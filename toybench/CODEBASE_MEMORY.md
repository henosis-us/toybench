# ToyBench Codebase Memory (Concise)

- Purpose: Lightweight agentic benchmark harness that runs LLMs on small, diverse tasks and records outcomes, artifacts, and metrics.
- Entry point: `toybench_cli.py` (CLI to select task, provider, model, attempts, and rounds; writes results under `results/`).
- Core flow: Load prompts → init environment → drive agent turns (single or conversational) → evaluate final outcome (deterministic or LLM-based) → aggregate metrics → save report + JSONL history.

## Tasks & Environments
- `file_system` → `environments/file_system_env.py`
  - Simulated POSIX-like FS with commands: `ls`, `cd`, `pwd`, `mkdir`, `cat`, `cp`, `rm`, `echo >`, `echo >>`.
  - Agent outputs one command per turn in a fenced block: ```action <cmd> ```; completes with `TASK_COMPLETE`.
  - Deterministic evaluation via prompt file `tasks/file_system/file_system_finaleval.txt` mapped to environment state.

- `tic_tac_toe` → `environments/tic_tac_toe_env.py`
  - Agent is `X`; opponent `O` plays optimally via Minimax.
  - Agent move format: `place X at r,c` in ```action``` block. Deterministic scoring (win/draw/invalid moves).

- `solar_gen` → `environments/solar_system_env.py`
  - Agent generates complete HTML each turn wrapped in `<solar.html>...</solar.html>` tags.
  - Uses Selenium to render HTML, capture screenshot, and browser logs; iterative refinement with intermediate feedback.
  - Final scoring is multimodal image evaluation (LLM) using screenshot and `tasks/solar_gen/solar_gen_finaleval.txt` rubric.

- `sokoban` → `environments/sokoban_env.py`
  - Puzzle pushing game on an ASCII grid (walls `#`, goals `.`, boxes `$`/`*`, player `@`/`+`).
  - Valid moves per turn: `up`, `down`, `left`, `right`, `reset`; agent outputs exactly one move inside ```action```.
  - Deterministic evaluation in the environment: 3 if all goals have boxes, 2 if some boxes on goals, 1 otherwise.
  - Visual artifacts per turn saved to attempt dir: `sokoban_iteration_<n>.html`, `sokoban_screenshot_iteration_<n>.png`, and logs. Screenshots are optional if Selenium isn’t available.
  - CLI: `--sokoban_level` selects the level index (0=base; several bundled levels include simple two‑push, easy, medium, and hard variants). Use `--list_sokoban_levels` to print all levels with brief notes.
  - Prompts: `tasks/sokoban/sokoban_goal.txt`, `tasks/sokoban/sokoban_generate.txt` (finaleval file exists for docs, not used).

## Prompts & Task Files
- Located under `tasks/<task_name>/` with four templates (some tasks omit intermediate or final):
  - `<task>_goal.txt`, `<task>_generate.txt`, `<task>_intermediate_eval.txt`, `<task>_finaleval.txt`.
- Key conventions enforced by prompts:
  - File System: commands must be inside ```action``` fences; output exactly one command per turn; finish with `TASK_COMPLETE`.
  - Solar: entire HTML must be emitted every turn strictly between `<solar.html>` and `</solar.html>`.
  - Sokoban: output exactly one move word per turn in ```action```; no prose.

## LLM Providers (in `llm_interface.py`)
- Supported: Google Gemini, OpenAI, xAI Grok, Quality Compute (sim), Anthropic, Kimi (Moonshot), OpenRouter.
- Interfaces expose: `generate_action`, `generate_action_conversational`, `generate_content_multimodal`, `evaluate_outcome`.
- Extras: reasoning/usage token extraction; OpenAI background polling; Anthropic/Gemini thinking config; OpenRouter options.

## Config & API Keys (`config.py` / `.env`)
- Env vars consumed: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `QC_API_KEY`, `QUALITY_COMPUTE_URL`, `ANTHROPIC_API_KEY2`, `KIMI_API_KEY2`, `OPENROUTER_API_KEY`.
- Defaults: see `default_*_model` in `config.py`; evaluator model defaults to a Gemini flash variant.

## Outputs & Reporting
- Run directory: `results/<task>_<provider>_<model>_<timestamp>/` with per-attempt subfolders.
- Artifacts:
  - Always: `run.log`, `run_config.json`, `summary_report.txt`, `attempt_results.jsonl` (turn-by-turn history, token usage per turn if available).
  - Solar per turn: `solar_iteration_<n>.html`, `solar_screenshot_iteration_<n>.png`, `browser_logs_iteration_<n>.txt`.
- Metrics: `reporting.calculate_metrics` aggregates success/partial/fail, pass@1, basic pass@20, regressions, and token totals; includes Solar “Differentiated Score”.

## Important Dependencies
- General: `openai`, `google-genai`, `anthropic`, `requests`, `python-dotenv`.
- Solar: `selenium` + Chrome/Chromedriver (headless). Without Selenium, Solar env initialization fails early.

## Typical CLI Usage
- Example (File System): `python toybench_cli.py -t file_system -p gemini -m gemini-2.0-flash -n 3 -r 40`
- Example (Solar): `python toybench_cli.py -t solar_gen -p openai -m gpt-4o -n 3 -r 8 --evaluator_model gemini-1.5-flash-8b`

## Gotchas / Conventions
- Solar task fails if the `<solar.html>` wrapper tags are missing or incomplete.
- File System requires exact ```action``` fencing; any extra prose breaks parsing.
- Tic-Tac-Toe is turn-synchronous; invalid agent moves are tracked and can reduce final score.
- For mixed providers, evaluator provider is inferred from `--evaluator_model` unless overridden.

## Quick Mental Model
- `toybench_cli.py` wires together config → provider interfaces → environment → attempt loop → evaluator → reporting.
- Deterministic tasks (FS, TTT) score internally; Solar uses multimodal LLM evaluator over screenshots.

## Places to Update Over Time
- Add new tasks under `tasks/<name>` and a matching `environments/<name>_env.py`; wire in `get_environment` and parsing in `toybench_cli.py`.
- Extend providers or options in `llm_interface.py` and `config.py`.
- Evolve metrics/reporting in `reporting.py`.
