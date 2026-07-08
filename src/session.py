"""Session 管理 — 每次运行工作流时创建独立目录。

目录结构:
  output/{研究问题(整理后)}/
    ├── papers.db          ← SQLite 数据库
    ├── {paper_id}.pdf     ← 下载的 PDF
    ├── {paper_id}.md      ← 转换的 Markdown
    └── review.md          ← 最终综述（后续）
"""

import re
from pathlib import Path


def sanitize_filename(text: str, max_len: int = 60) -> str:
    """将文本整理为可用于文件/目录名的字符串。

    处理:
      - 去除特殊字符
      - 中文保留
      - 空格变下划线
      - 截断到 max_len
    """
    # 替换分隔符为空格
    text = re.sub(r"[/\\:：、，,，。.。?？!！" r"''（）()【】\[\]{}]", " ", text)
    # 多个空格合并
    text = re.sub(r"\s+", "_", text.strip())
    # 去除非法的文件名字符（保留中文、字母、数字、下划线、横线）
    text = re.sub(r"[^\w\u4e00-\u9fff_-]", "", text)
    if not text:
        text = "unnamed_session"
    return text[:max_len]


def create_session(question: str, base_dir: str = "output") -> Path:
    """根据研究问题创建 session 目录。

    Args:
        question: 研究问题（用于命名目录）。
        base_dir: 基础输出目录（默认 "output"）。

    Returns:
        session 目录的 Path 对象。
    """
    dir_name = sanitize_filename(question)
    session_path = Path(base_dir) / dir_name
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path.resolve()
