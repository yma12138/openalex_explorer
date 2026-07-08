---
name: openalex-explorer
description: Automated literature review pipeline: search academic papers → relevance screening → quality scoring → structured summary extraction → review writing, suitable for rapid exploration of academic fields, producing high-quality paper lists and reports. Analysis does not go through full paper text. If the user wants the AI to read all papers and summarize their content, do not use this skill.
---

## Input Configuration (Three JSON Files,If you Don't have these,you need to create them)

### `llm_config.json` — LLM Configuration (API key required)

```json
{"model": "deepseek/deepseek-chat", "api_key": "sk-...", "temperature": 0.3, "max_tokens": 4096}
```

| Field | Required | Description |
|---|---|---|
| `model` | ✅ | litellm format `provider/model-name`, e.g., `deepseek/deepseek-chat`, `openai/gpt-4o` |
| `api_key` | ✅ | Core credential for the pipeline |
| `temperature` | ❌ | Default 0.3, range 0–2 |
| `max_tokens` | ❌ | Default 4096 |

Additional parameters can be passed via `_extra` (e.g., `api_base`, `api_version`).

### `openalex_config.json` — OpenAlex Configuration (strongly recommended)

```json
{"mailto": "your-email@example.com", "api_key": "your-openalex-api-key"}
```

| Field | Required | Description |
|---|---|---|
| `mailto` | ❌ | If provided, rate limit increases from ~10 req/s to ~100 req/s |
| `api_key` | ❌ | Freely obtainable at openalex.org |

OpenAlex is completely free. Not providing it will **work**, but rate limiting will be strict and the experience will be poor.

### Spec JSON — Research Question Description (Core Input)

```json
{
  "research_question": "A Survey on Methods for Improving Reasoning Capabilities of Large Language Models",
  "start_year": 2022, "end_year": 2026, "min_papers": 15,
  "method_preference": "chain-of-thought, tree-of-thought",
  "research_field": "natural language processing",
  "initial_depth": 20, "max_total_papers": 500, "max_quality_papers": 30,
  "relevance_workers": 20, "quality_workers": 20, "max_refine_rounds": 3,
  "language": "en", "lang_instruction": "All output should be in English."
}
```

**Research Question and Scope:**

| Field | Required | Default | Description |
|---|---|---|---|
| `research_question` | ✅ | — | The problem you want to research, need to be detailed, must be non‑empty |
| `start_year` / `end_year` | ❌ | 0 | Year range (0 = no limit) |
| `min_papers` | ❌ | 15 | Target number of quality_passed papers |
| `method_preference` | ❌ | "" | Preferred methodologies, e.g., "chain-of-thought" |
| `research_field` | ❌ | "" | Domain restriction, e.g., "natural language processing" |

**Pipeline Parameters:**

| Field | Default | Description |
|---|---|---|
| `initial_depth` | 20 | Initial search count per keyword |
| `max_total_papers` | 1000 | Hard database limit; stops all searches immediately when reached |
| `max_quality_papers` | 200 | Limit for quality_passed papers; stops searches when reached |
| `relevance_workers` | 20 | Concurrency for relevance screening |
| `quality_workers` | 20 | Concurrency for quality screening |
| `max_refine_rounds` | 3 | Maximum rounds for keyword back‑refinement |

**Language Control:**

| Field | Default | Description |
|---|---|---|
| `language` | `"en"` | Prompt language, `zh‑hans` or `en` |
| `lang_instruction` | `""` | Free‑form text prepended to every system prompt. Used to enforce output language |

When `lang_instruction` is non‑empty, each system prompt becomes `lang_instruction + "\n\n" + load_prompt(name, lang)`. The two fields work best together.

---

## Core Scripts

### `build_db.py` — Build the Database

```bash
python3 build_db.py <spec.json> [--llm-config PATH] [--spec JSON] [--output-dir DIR]
```

6‑step process:
1. LLM generates 7+ sets of search keywords
2. Warm‑up search of 5 papers per keyword to understand the field
3. Analyze warm‑up results and decide whether to adjust keyword directions
4. Full `depth` search with an async pipeline: search → relevance screening → quality screening
5. If quality_passed is insufficient, LLM back‑refines new keywords, up to `max_refine_rounds` rounds
6. Extract structured summaries for all quality_passed papers

Output: `output/{question}_{timestamp}/papers.db` + `session.json`

Error handling: invalid API key → exit with error | network issues → HTTP error messages | insufficient papers → warning and return with existing results.

### `write_review.py` — Write the Review

```bash
python3 write_review.py <session_dir_or_db> [--llm-config PATH] [--lang zh-hans|en] [--lang-instruction TEXT] [--question TEXT]
```

5‑step process:
1. Assign papers to 3–5 topics
2. Within each topic, analyse paper relationships in parallel (contrast/continuation/complement/independence)
3. Write chapter for each topic in parallel
4. Write Introduction + Conclusion + Abstract
5. Validate citation integrity

Prerequisite: `papers.db` exists and has at least 1 `quality_passed` paper.

### `manage.py` — Project Management

```bash
python3 manage.py list              # List projects (reverse chronological)
python3 manage.py show <index|path> # Show details
python3 manage.py delete <index|path> # Delete (with confirmation)
python3 manage.py export <index|path> # Export to Excel
python3 manage.py clean-secrets     # Clear all API keys
```

`list` example:
```
  #   Question                         Date             Papers Passed Review
─── ─────────────────────────────────────────────────────────────────
  1  A Survey on Methods for Improving  2026-07-08 00:56    146     65
     Reasoning Capabilities of LLMs
  2  A Survey on Smart Grid Development 2026-07-07 23:29  100     48   ✅
```

`export` outputs 15 columns, colour‑coded by status, with frozen header row + auto‑filter.

`clean-secrets` clears keys in `llm_config.json`, `openalex_config.json`, and all `session.json` files.

---

## Token Usage

Automatically recorded in `session.json`:

```json
{"build_usage": {"calls": 219, "total_tokens": 122954}, "review_usage": {"calls": 32, "total_tokens": 77854}}
```

Use `manage.py show` to view directly.

---

## Directory Structure

```
├── llm_config.json            ← LLM config (gitignored)
├── openalex_config.json       ← OpenAlex config (gitignored)
├── specs/*.json               ← Research question descriptions
├── build_db.py                ← Database builder
├── write_review.py            ← Review writer
├── manage.py                  ← Project manager
├── config/settings.yaml       ← Pipeline parameters
├── config/languages.json      ← Language mapping
├── src/
│   ├── llm.py                 ← litellm wrapper
│   ├── pipeline_async.py      ← Async queue pipeline
│   ├── filters.py             ← Relevance/quality filters
│   ├── keywords.py            ← Keyword brainstorming
│   ├── summarizer.py          ← Structured summary extraction
│   ├── prompts.py             ← prompts.json loader
│   ├── agents/prompts.json    ← 72 prompt templates
│   ├── review/                ← Review orchestration + data models
│   ├── store.py               ← SQLite storage
│   ├── session.py             ← Session directory management
│   ├── sources/openalex.py    ← OpenAlex search source
│   └── ...                    ← Other utility modules
└── output/                    ← Project outputs (gitignored)
```

---

## Frequently Asked Questions

**Q: Where do I put the API keys?**
A: LLM key → `api_key` in `llm_config.json`. OpenAlex key → `api_key` in `openalex_config.json`.

**Q: What is the difference between `language` and `lang_instruction`?**
A: `language` selects the prompt template language. `lang_instruction` is free‑form text prepended to every system prompt. They work together: `language` provides the correct prompt context, `lang_instruction` further enforces output language. If the user wants to use a language other than Chinese or English, set `language` to `en` and pass the specific language output instruction via `lang_instruction`.

**Q: Which models are supported?**
A: All litellm‑supported models: `deepseek/deepseek-chat`, `openai/gpt-4o`, `anthropic/claude-3-opus`, `google/gemini-pro`, etc.

**Q: What if the user's instructions are ambiguous and I cannot determine every JSON parameter?**
A: Call a question‑asking tool or similar to confirm with the user.

**Q: I need to know what Python environment is required to run this skill.**
A: Please refer to pyproject.toml.