#!/usr/bin/env python3
"""最终分析 — 从已有数据库读取论文并撰写综述。

用法:
  python write_review.py output/session_name/paper_store.db --llm-config llm_config.json
  python write_review.py output/session_name/                     # 自动找 papers.db
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.http_client import HttpClient
from src.llm import LLM
from src.review.orchestrator import ReviewOrchestrator
from src.store import PaperStore
from src.topic import TopicSpec


async def write(
    db_path: Path,
    llm_config: dict,
    lang: str = "zh-hans",
    lang_instruction: str = "",
    question: str = "",
) -> str:
    """从已有数据库撰写综述，返回 review.md 路径。"""
    if not llm_config.get("api_key"):
        print("❌ LLM config missing api_key")
        sys.exit(1)

    store = PaperStore(str(db_path))
    quality_count = store.count(status="quality_passed")

    if quality_count == 0:
        print("❌ No quality_passed papers in DB")
        store.close()
        sys.exit(1)

    print(f"  Papers: {quality_count} quality_passed")

    # 从 DB 读取研究的 research question
    # 从第一条 quality_passed 论文找 search_query 信息
    papers = store.list_papers(status="quality_passed", limit=1)
    if not papers:
        papers = store.list_papers(limit=1)

    # 如果没传 question，尝试从 DB 文件名所在目录提取
    spec_question = question or db_path.parent.name.split("_202")[0].replace("_", " ")

    spec = TopicSpec(research_question=spec_question)
    session_dir = db_path.parent

    await HttpClient.init(max_concurrent=100)
    llm = LLM(config=llm_config)

    print(f"\n{'=' * 60}")
    print("  Writing review")
    print(f"{'=' * 60}")

    review_worker = ReviewOrchestrator(
        llm=llm,
        store=store,
        spec=spec,
        output_dir=session_dir,
        lang=lang,
        lang_instruction=lang_instruction,
    )

    draft = await review_worker.run()
    store.close()
    await HttpClient.close()

    if not draft:
        print("❌ Review draft empty")
        sys.exit(1)

    review_path = str(session_dir / "review.md")
    print(f"\n✅ Review saved: {review_path}")
    print(f"   Sections: {len(draft.sections)}")
    words = sum(len(s.content) for s in draft.sections) + len(draft.introduction) + len(draft.conclusion)
    print(f"   Words: {words}")

    # 保存 token 用量到 session.json
    usage = llm.get_usage()
    session_json = session_dir / "session.json"
    if session_json.exists():
        with open(session_json) as f:
            info = json.load(f)
    else:
        info = {}
    info["review_usage"] = usage
    info["review_stats"] = {"sections": len(draft.sections), "words": words}
    with open(session_json, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  Tokens: {usage['total_tokens']:,} ({usage['calls']} calls)")

    return review_path


def main():
    parser = argparse.ArgumentParser(description="从已有数据库撰写综述")
    parser.add_argument("session", help="Session 目录或 papers.db 路径")
    parser.add_argument("--llm-config", default="llm_config.json")
    parser.add_argument("--lang", default="zh-hans", choices=["zh-hans", "en"])
    parser.add_argument("--lang-instruction", default="")
    parser.add_argument("--question", default="", help="研究问题（可选，自动从目录名推断）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 定位 DB
    session_path = Path(args.session)
    if session_path.is_dir():
        db_path = session_path / "papers.db"
    elif session_path.name == "papers.db":
        db_path = session_path
    else:
        db_path = session_path  # 让用户尝试

    if not db_path.exists():
        print(f"❌ DB not found: {db_path}")
        sys.exit(1)

    # LLM config
    llm_path = Path(args.llm_config)
    if not llm_path.exists():
        print(f"❌ LLM config not found: {llm_path}")
        sys.exit(1)
    with open(llm_path) as f:
        llm_config = json.load(f)

    asyncio.run(
        write(
            db_path,
            llm_config,
            lang=args.lang,
            lang_instruction=args.lang_instruction,
            question=args.question,
        )
    )


if __name__ == "__main__":
    main()
