# OpenAlex Explorer

An [OpenCode](https://github.com/opencode-ai/opencode) skill for automated literature review.

## How It Works

After registering this skill, you simply **tell your AI agent what you want to research**, and it handles the rest — searching academic papers, screening for relevance and quality, extracting summaries, and writing a structured review.

The pipeline:

1. **Keyword Generation** — AI brainstorms search keywords from your question
2. **Adaptive Search** — Searches [OpenAlex](https://openalex.org/) with async parallelism
3. **Relevance & Quality Screening** — Two-stage LLM filtering
4. **Structured Summaries** — Extracts key findings
5. **Review Writing** — Topic clustering → parallel chapter writing → citation validation

## Quick Start

Just tell your AI agent something like:

> *"I want to do a literature review on improving reasoning in LLMs, covering 2022-2026, at least 15 papers, in English."*

The agent will read the skill definition, ask you for any missing details, configure the JSON spec, and run the full pipeline automatically.

## Manual CLI Usage (Optional)

You can also run the pipeline directly:

```bash
# Build the paper database
python3 build_db.py specs/my_review.json

# Write the review
python3 write_review.py output/My_Review_20260708_120000/

# Manage sessions
python3 manage.py list
python3 manage.py export 0
```

## Configuration Reference

### `llm_config.json`

| Field | Required | Description |
|---|---|---|
| `model` | ✅ | `provider/model-name` (e.g., `deepseek/deepseek-chat`, `openai/gpt-4o`) |
| `api_key` | ✅ | API credential |
| `temperature` | ❌ | Default 0.3 |
| `max_tokens` | ❌ | Default 4096 |

### `openalex_config.json`

| Field | Required | Description |
|---|---|---|
| `mailto` | ❌ | Email for polite pool (~100 req/s) |
| `api_key` | ❌ | Free at openalex.org |

### Spec JSON (Research Question)

| Field | Required | Default | Description |
|---|---|---|---|
| `research_question` | ✅ | — | Your research topic |
| `start_year` / `end_year` | ❌ | 0 | Year range |
| `min_papers` | ❌ | 15 | Target papers |
| `method_preference` | ❌ | "" | Preferred methods |
| `research_field` | ❌ | "" | Domain restriction |
| `initial_depth` | ❌ | 20 | Search depth per keyword |
| `max_total_papers` | ❌ | 1000 | Hard database limit |
| `max_quality_papers` | ❌ | 200 | Quality-passed limit |
| `relevance_workers` | ❌ | 20 | Concurrency |
| `quality_workers` | ❌ | 20 | Concurrency |
| `language` | ❌ | `"en"` | `"en"` or `"zh-hans"` |
| `lang_instruction` | ❌ | `""` | Extra prompt prefix |

## Project Structure

```
├── llm_config.json            ← LLM config (gitignored)
├── openalex_config.json       ← OpenAlex config (gitignored)
├── specs/*.json               ← Research specs (created by agent)
├── SKILL.md                   ← Skill definition
├── build_db.py                ← Database builder
├── write_review.py            ← Review writer
├── manage.py                  ← Session manager
├── src/                       ← Core modules
└── output/                    ← Results (gitignored)
```

## Requirements

- Python >= 3.12
- An LLM API key
- OpenAlex API KEY,it's free

## License

MIT
