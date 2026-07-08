#!/usr/bin/env python3
"""建立数据库 — 搜索 → 相关性 → 质量 → 摘要提取。

用法:
  python build_db.py specs/test_input.json --llm-config llm_config.json
  python build_db.py --spec '{"research_question":"...","min_papers":15}'
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.http_client import HttpClient
from src.llm import LLM
from src.pipeline_async import PipelineConfig, PipelineOrchestrator
from src.session import sanitize_filename
from src.store import PaperStore
from src.topic import TopicSpec


# ── 复用小部分解析逻辑 ─────────────────────────────────


def _parse_spec(source: str) -> dict:
    path = Path(source)
    if path.exists() and source.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    return json.loads(source)


def _validate(data: dict) -> list[str]:
    errors = []
    if not data.get("research_question", "").strip():
        errors.append("research_question cannot be empty")
    mp = data.get("min_papers", 15)
    if mp < 1:
        errors.append("min_papers cannot be less than 1")
    if mp > 500:
        errors.append("min_papers cannot exceed 500")
    if data.get("start_year") and data.get("end_year") and data["start_year"] > data["end_year"]:
        errors.append("start_year cannot exceed end_year")
    return errors


# ── 主流程 ─────────────────────────────────────────────


async def build(
    spec_data: dict,
    llm_config: dict,
    output_dir: str = "output",
) -> str:
    """运行搜索 ~ 摘要流水线，返回 session 目录路径。"""
    if not llm_config.get("api_key"):
        print("❌ LLM config missing api_key")
        sys.exit(1)

    await HttpClient.init(max_concurrent=100)
    llm = LLM(config=llm_config)

    # Session 目录
    ts = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{sanitize_filename(spec_data['research_question'])}_{ts}"
    session_dir = Path(output_dir) / dir_name
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Session: {session_dir}")

    # TopicSpec
    spec = TopicSpec(
        research_question=spec_data["research_question"],
        start_year=spec_data.get("start_year", 0),
        end_year=spec_data.get("end_year", 0),
        min_papers=spec_data.get("min_papers", 15),
        method_preference=spec_data.get("method_preference", ""),
        research_field=spec_data.get("research_field", ""),
    )

    # PipelineConfig
    pipe_cfg = PipelineConfig(
        initial_depth=spec_data.get("initial_depth", 20),
        max_total_papers=spec_data.get("max_total_papers", 1000),
        max_quality_papers=spec_data.get("max_quality_papers", 200),
        relevance_workers=spec_data.get("relevance_workers", 20),
        quality_workers=spec_data.get("quality_workers", 20),
        max_refine_rounds=spec_data.get("max_refine_rounds", 3),
        language=spec_data.get("language", "zh-hans"),
        lang_instruction=spec_data.get("lang_instruction", ""),
    )

    # 运行流水线
    orchestrator = PipelineOrchestrator(llm=llm, spec=spec, config=pipe_cfg)
    old_dir = orchestrator.session_dir
    shutil.rmtree(str(old_dir), ignore_errors=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    orchestrator.session_dir = session_dir
    orchestrator.store = PaperStore(str(session_dir / "papers.db"))

    stats = await orchestrator.run_full()

    store = orchestrator.store
    print(
        f"\n{'=' * 50}\n"
        f"  Done: searched={stats.total_searched}  relevanced={stats.relevanced}\n"
        f"  quality_passed={stats.quality_passed}  rejected={stats.rejected}\n"
        f"  refine rounds={stats.refine_rounds}"
    )

    store.close()
    await HttpClient.close()

    # 保存 session 元信息（含 token 用量）
    usage = llm.get_usage()
    session_info = {
        "research_question": spec_data["research_question"],
        "pipeline_stats": {
            "searched": stats.total_searched,
            "relevanced": stats.relevanced,
            "quality_passed": stats.quality_passed,
            "rejected": stats.rejected,
            "refine_rounds": stats.refine_rounds,
        },
        "build_usage": usage,
    }
    with open(session_dir / "session.json", "w", encoding="utf-8") as f:
        json.dump(session_info, f, ensure_ascii=False, indent=2)
    print(f"  Tokens: {usage['total_tokens']:,} ({usage['calls']} calls)")

    return str(session_dir)


# ── CLI ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="建立数据库 — 搜索 → 质量 → 摘要")
    parser.add_argument("input", nargs="?", help="JSON 文件路径或 JSON 字符串")
    parser.add_argument("--spec", help="JSON 字符串（替代位置参数）")
    parser.add_argument("--llm-config", default="llm_config.json",
                        help="LLM 配置 JSON 文件路径")
    parser.add_argument("--output-dir", default="output",
                        help="输出目录（默认 output/）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    source = args.spec or args.input
    if not source:
        if not sys.stdin.isatty():
            source = sys.stdin.read().strip()
    if not source:
        parser.print_help()
        sys.exit(1)

    try:
        data = _parse_spec(source)
    except Exception as e:
        print(f"❌ Spec parsing failed: {e}")
        sys.exit(1)

    errors = _validate(data)
    if errors:
        for e in errors:
            print(f"❌ {e}")
        sys.exit(1)

    # LLM config
    llm_path = Path(args.llm_config)
    if not llm_path.exists():
        print(f"❌ LLM config not found: {llm_path}")
        sys.exit(1)
    with open(llm_path) as f:
        llm_config = json.load(f)

    session_dir = asyncio.run(build(data, llm_config, args.output_dir))
    print(f"\n✅ Session: {session_dir}")
    print(f"   DB: {session_dir}/papers.db")


if __name__ == "__main__":
    main()
