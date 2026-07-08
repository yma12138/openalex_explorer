"""Utility functions for the literature review agent."""

import re
from pathlib import Path
from typing import Optional


def extract_toc(md_path: str | Path) -> Optional[str]:
    """Extract table of contents from a Markdown file by parsing headers.

    Parses all markdown headings (# through ######) and returns
    them as an indented tree structure.

    Args:
        md_path: Path to the Markdown file.

    Returns:
        Formatted table of contents string, or None if file not found.
    """
    path = Path(md_path)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    toc_lines = []
    for line in lines:
        stripped = line.rstrip()
        # Match markdown headers: # through ######
        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            # Remove bold/italic markers for cleaner display
            title = re.sub(r"[\*_]{1,2}", "", title).strip()
            indent = "  " * (level - 1)
            toc_lines.append(f"{indent}{'│ ' if level > 1 else ''}{title}")

    if not toc_lines:
        return "(no headers found)"

    result = "\n".join(toc_lines)
    return result


def count_chars(md_path: str | Path) -> Optional[dict]:
    """Count characters and lines in a Markdown file."""
    path = Path(md_path)
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    return {
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "words": len(text.split()),
    }
