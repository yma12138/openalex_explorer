#!/usr/bin/env python3
"""Skill 入口 — 接收 TopicSpec JSON，运行完整流水线，输出结果。

用法:
    python3 -m src.skill specs/test_input.json
    python3 -m src.skill --spec '{"research_question":"...","min_papers":10}'
    python3 -m src.skill --dry-run specs/test_input.json   # 仅验证输入
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.http_client import HttpClient
from src.llm import LLM
from src.pipeline_async import PipelineConfig, PipelineOrchestrator
from src.review.orchestrator import ReviewOrchestrator
from src.session import sanitize_filename
from src.store import PaperStore
from src.topic import TopicSpec

# ══════════════════════════════════════════════════════════
# Skill 输入 / 输出模型
# ══════════════════════════════════════════════════════════


@dataclass
class SkillInput:
    """Skill 输入参数 — 由外部传入 JSON 或文件。"""

    # 必填
    research_question: str

    # 搜索范围
    start_year: int = 0
    end_year: int = 0
    min_papers: int = 15

    # 领域限定（可选）
    method_preference: str = ""
    research_field: str = ""

    # 流水线参数
    initial_depth: int = 20
    max_total_papers: int = 1000
    max_quality_papers: int = 200
    relevance_workers: int = 20
    quality_workers: int = 20
    max_refine_rounds: int = 3

    # 语言
    language: str = "zh-hans"  # "zh-hans" 简体中文, "zh-tw" 繁体中文, "en" English
    lang_instruction: str = ""  # 可选：自由传入的语言指令字符串，会追加到每条系统提示词前

    # 输出
    output_dir: str = "output"
    generate_review: bool = True


@dataclass
class SkillOutput:
    """Skill 输出结果。"""

    success: bool = False
    session_dir: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    review_path: str = ""
    error: str = ""
    elapsed: float = 0.0


# ══════════════════════════════════════════════════════════
# 输入解析
# ══════════════════════════════════════════════════════════


def parse_input(source: str) -> SkillInput:
    """从 JSON 字符串或文件路径解析 SkillInput。

    Args:
        source: JSON 字符串或以 .json 结尾的文件路径。

    Returns:
        SkillInput 实例。
    """
    path = Path(source)
    if path.exists() and source.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(source)

    return SkillInput(
        research_question=data.get("research_question", ""),
        start_year=data.get("start_year", 0),
        end_year=data.get("end_year", 0),
        min_papers=data.get("min_papers", 15),
        method_preference=data.get("method_preference", ""),
        research_field=data.get("research_field", ""),
        initial_depth=data.get("initial_depth", 20),
        max_total_papers=data.get("max_total_papers", 1000),
        max_quality_papers=data.get("max_quality_papers", 200),
        relevance_workers=data.get("relevance_workers", 20),
        quality_workers=data.get("quality_workers", 20),
        max_refine_rounds=data.get("max_refine_rounds", 3),
        output_dir=data.get("output_dir", "output"),
        generate_review=data.get("generate_review", True),
        language=data.get("language", "zh-hans"),
        lang_instruction=data.get("lang_instruction", ""),
    )


def validate_input(inp: SkillInput) -> list[str]:
    """校验输入参数，返回错误列表。"""
    errors = []
    if not inp.research_question.strip():
        errors.append("research_question cannot be empty")
    if inp.min_papers < 1:
        errors.append("min_papers cannot be less than 1")
    if inp.min_papers > 500:
        errors.append("min_papers cannot exceed 500")
    if inp.start_year and inp.end_year and inp.start_year > inp.end_year:
        errors.append("start_year cannot exceed end_year")
    if inp.initial_depth < 1:
        errors.append("initial_depth cannot be less than 1")
    return errors


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════


async def run_skill(inp: SkillInput, llm_config: dict) -> SkillOutput:
    """运行完整技能流水线。

    流程:
      1. 创建 session 目录
      2. 搜索 → 相关性 → 质量 流水线
      3. PDF 下载
      4. Writing review（可选）
      5. 返回结果
    """
    start = time.time()
    output = SkillOutput()

    # 配置检查
    if not llm_config.get("api_key"):
        return SkillOutput(success=False, error="LLM config missing api_key")

    await HttpClient.init(max_concurrent=100)
    llm = LLM(config=llm_config)

    # ── 1. Session 目录 ────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{sanitize_filename(inp.research_question)}_{ts}"
    session_dir = Path(inp.output_dir) / dir_name
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Session: {session_dir}")

    # ── 2. TopicSpec ───────────────────────────────────
    spec = TopicSpec(
        research_question=inp.research_question,
        start_year=inp.start_year,
        end_year=inp.end_year,
        min_papers=inp.min_papers,
        method_preference=inp.method_preference,
        research_field=inp.research_field,
    )

    # ── 3. 流水线配置 ──────────────────────────────────
    pipe_cfg = PipelineConfig(
        initial_depth=inp.initial_depth,
        max_total_papers=inp.max_total_papers,
        max_quality_papers=inp.max_quality_papers,
        relevance_workers=inp.relevance_workers,
        quality_workers=inp.quality_workers,
        max_refine_rounds=inp.max_refine_rounds,
        language=inp.language,
        lang_instruction=inp.lang_instruction,
    )

    # ── 4. 搜索流水线 ──────────────────────────────────
    orchestrator = PipelineOrchestrator(llm=llm, spec=spec, config=pipe_cfg)
    # 接管 session 目录
    import shutil

    old_dir = orchestrator.session_dir
    shutil.rmtree(str(old_dir), ignore_errors=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    orchestrator.session_dir = session_dir
    orchestrator.store = PaperStore(str(session_dir / "papers.db"))

    pipeline_stats = await orchestrator.run_full()
    store = orchestrator.store

    output.stats = {
        "total_searched": pipeline_stats.total_searched,
        "relevanced": pipeline_stats.relevanced,
        "quality_passed": pipeline_stats.quality_passed,
        "rejected": pipeline_stats.rejected,
        "refine_rounds": pipeline_stats.refine_rounds,
    }

    # ── 5. Writing review（可选） ────────────────────────────
    review_path = ""
    if inp.generate_review and store.count(status="quality_passed") > 0:
        print(f"\n{'=' * 60}")
        print("  Writing review")
        print(f"{'=' * 60}")
        review_worker = ReviewOrchestrator(
            llm=llm,
            store=store,
            spec=spec,
            output_dir=session_dir,
            lang=inp.language,
            lang_instruction=inp.lang_instruction,
        )
        draft = await review_worker.run()
        if draft:
            review_path = str(session_dir / "review.md")
            output.stats["review_sections"] = len(draft.sections)
            output.stats["review_words"] = (
                sum(len(s.content) for s in draft.sections)
                + len(draft.introduction)
                + len(draft.conclusion)
            )

    store.close()
    await HttpClient.close()

    elapsed = time.time() - start
    output.success = True
    output.session_dir = str(session_dir)
    output.review_path = review_path
    output.elapsed = round(elapsed, 1)

    return output


# ══════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(description="文献综述 Agent Skill")
    parser.add_argument("input", nargs="?", help="JSON 文件路径或 JSON 字符串")
    parser.add_argument("--spec", help="JSON 字符串（替代位置参数）")
    parser.add_argument("--llm-config", default="llm_config.json",
                        help="LLM 配置 JSON 文件路径（默认 llm_config.json）")
    parser.add_argument("--dry-run", action="store_true", help="仅校验输入")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 获取输入
    source = args.spec or args.input
    if not source:
        # 尝试从 stdin 读取
        if not sys.stdin.isatty():
            source = sys.stdin.read().strip()
    if not source:
        parser.print_help()
        sys.exit(1)

    # 解析
    try:
        inp = parse_input(source)
    except Exception as e:
        print(f"❌ Input parsing failed: {e}")
        sys.exit(1)

    # 校验
    errors = validate_input(inp)
    if errors:
        for e in errors:
            print(f"❌ {e}")
        sys.exit(1)

    print(f"✅ Input validation passed: {inp.research_question[:50]}...")
    print(
        f"   Year range: {inp.start_year}-{inp.end_year}  Min papers: {inp.min_papers}"
    )

    if args.dry_run:
        print("🏁 Dry-run mode, no pipeline execution")
        return

    # 加载 LLM 配置
    llm_config_path = Path(args.llm_config)
    if not llm_config_path.exists():
        print(f"❌ LLM config not found: {llm_config_path}")
        sys.exit(1)
    with open(llm_config_path, "r", encoding="utf-8") as f:
        llm_config = json.load(f)

    # 运行
    output = await run_skill(inp, llm_config)

    # 输出结果
    print(f"\n{'=' * 60}")
    print(f"  Skill completed ({output.elapsed}s)")
    print(f"{'=' * 60}")
    print(
        json.dumps(
            {
                "success": output.success,
                "session_dir": output.session_dir,
                "review_path": output.review_path,
                "stats": output.stats,
                "elapsed": output.elapsed,
                "error": output.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
