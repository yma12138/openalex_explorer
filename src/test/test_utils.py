"""Tests for utility functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import count_chars, extract_toc

SAMPLE_MD = """# Title

Some intro text.

## Section 1

Content here.

### Subsection 1.1

More content.

## Section 2

Final part.
"""


def test_extract_toc(tmp_path):
    md_file = tmp_path / "sample.md"
    md_file.write_text(SAMPLE_MD, encoding="utf-8")

    toc = extract_toc(md_file)
    assert toc is not None
    assert "Title" in toc
    assert "Section 1" in toc
    assert "Subsection 1.1" in toc
    assert "Section 2" in toc


def test_extract_toc_file_not_found():
    result = extract_toc("/nonexistent/file.md")
    assert result is None


def test_extract_toc_empty_file(tmp_path):
    md_file = tmp_path / "empty.md"
    md_file.write_text("", encoding="utf-8")
    result = extract_toc(md_file)
    assert result == "(no headers found)"


def test_extract_toc_strips_formatting(tmp_path):
    md = "# **Bold Title**\n\n## *Italic Section*\n"
    md_file = tmp_path / "fmt.md"
    md_file.write_text(md, encoding="utf-8")

    toc = extract_toc(md_file)
    assert "**" not in toc
    assert "*" not in toc
    assert "Bold Title" in toc
    assert "Italic Section" in toc


def test_count_chars(tmp_path):
    md_file = tmp_path / "stats.md"
    md_file.write_text("Hello world!\n\nLine 3.", encoding="utf-8")

    stats = count_chars(md_file)
    assert stats is not None
    assert stats["chars"] == 21
    assert stats["lines"] == 3
    assert stats["words"] == 4


def test_count_chars_file_not_found():
    result = count_chars("/nonexistent/file.md")
    assert result is None
