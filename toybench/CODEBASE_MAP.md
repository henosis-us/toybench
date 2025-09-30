ToyBench Codebase Map

What this is
- Lightweight agentic benchmark harness: runs LLMs on small, diverse interactive tasks and records outcomes, artifacts, and metrics.
- Run via a single CLI; supports multiple model providers; saves results under results/.

Entry point
- toybench_cli.py — CLI to select task, provider/model, attempts, rounds, evaluator model, and provider-specific options.

Top-level layout
- environments/ — task environments (deterministic game/sim + orchestration)
  - base_env.py — abstract BaseEnvironment interface
  - file_system_env.py — simulated POSIX-like FS with ls/cd/pwd/mkdir/cat/cp/rm/echo >, echo >>; complex goal; deterministic eval
  - tic_tac_toe_env.py — agent is X vs optimal O (Minimax); one move per turn; deterministic eval
  - solar_system_env.py — iterative HTML generation; renders with Selenium; captures screenshot/logs; intermediate + final LLM eval
  - sokoban_env.py — ASCII Sokoban with bundled levels; per-turn HTML/screenshot artifacts; deterministic eval
- tasks/ — prompt templates per task
  - <task>_goal.txt, <task>_generate.txt, <task>_intermediate_eval.txt (optional), <task>_finaleval.txt
- llm_interface.py — provider adapters (Gemini, OpenAI, xAI Grok, Quality Compute, Anthropic, Kimi, OpenRouter) exposing
  - generate_action, generate_action_conversational, generate_content_multimodal, evaluate_outcome
- evaluation.py — Evaluator wrapper for final scoring
  - Text eval: formats final prompt; parses <rating>1..3</rating>
  - Image eval (solar): sends screenshot + rubric to multimodal model
- reporting.py — aggregates metrics, token usage; formats/saves report and JSONL history
- browser_utils.py — Selenium rendering, screenshot capture, console log harvesting, image encoding for multimodal
- config.py — loads API keys and defaults from .env; selects default models
- utils.py — logging, output dir creation, score parsing
- tests/ — basic smoke tests (e.g., Sokoban)
- results/ — run outputs (created by CLI)

How a run works (simplified)
- CLI parses args → loads config/keys → loads task prompts → instantiates provider interfaces → builds environment → loops turns:
  - Environment provides prompt context → agent LLM proposes an action → CLI parses and validates action → env.step returns feedback/new state
  - Stop on terminal state, error, or step budget; then evaluate (deterministic or via LLM) → report + save artifacts

Tasks at a glance
- file_system
  - One command per turn inside ```action ...```; finish with TASK_COMPLETE.
  - Deterministic scoring via environment (checks dirs/files/contents and final cwd).
- tic_tac_toe
  - Agent outputs coordinates (e.g., "place X at r,c" or "r,c" parsed); opponent O plays optimally via Minimax.
  - Deterministic scoring; invalid moves penalize.
- solar_gen
  - Agent must emit full HTML each turn between <solar.html>...</solar.html> tags.
  - Renders with headless Chrome → screenshot → intermediate LLM feedback → refine for fixed number of rounds.
  - Final score via multimodal evaluator over the screenshot using tasks/solar_gen/solar_gen_finaleval.txt.
- sokoban
  - Actions: up/down/left/right/reset; artifacts (HTML, optional screenshot/logs) saved per turn.
  - Deterministic scoring: 3 solved, 2 partial (some boxes on goals), 1 otherwise; supports --sokoban_level and --list_sokoban_levels.

Providers (llm_interface.py)
- Gemini, OpenAI (/v1/responses + optional background polling), xAI Grok, Quality Compute (best-of-N or collaborative), Anthropic (thinking), Kimi (Moonshot), OpenRouter (routing/fallbacks provider selection).
- Token usage captured when available; reasoning/thinking tokens supported for some providers.

Outputs (per run under results/<task>_<provider>_<model>_<ts>/)
- run.log, run_config.json, summary_report.txt, attempt_results.jsonl (turn-by-turn with token usage if available)
- solar_gen: solar_iteration_<n>.html, solar_screenshot_iteration_<n>.png, browser_logs_iteration_<n>.txt
- sokoban: sokoban_iteration_<n>.html, sokoban_screenshot_iteration_<n>.png (if Selenium available), sokoban_browser_logs_iteration_<n>.txt

Quickstart
- Requirements: Python 3.10+, install deps
  - pip install -r requirements.txt
  - For providers: pip install openai anthropic (and others as needed)
  - For solar_gen artifacts: install Selenium + Chrome/Chromedriver and ensure chromedriver is in PATH
- Set API keys in .env
  - GOOGLE_API_KEY, OPENAI_API_KEY, XAI_API_KEY, QC_API_KEY, QUALITY_COMPUTE_URL, ANTHROPIC_API_KEY2, KIMI_API_KEY2, OPENROUTER_API_KEY
- Example runs
  - File System: python toybench_cli.py -t file_system -p gemini -m gemini-2.0-flash -n 3 -r 40
  - Tic-Tac-Toe: python toybench_cli.py -t tic_tac_toe -p openai -m o4-mini -n 5
  - Solar (needs Selenium): python toybench_cli.py -t solar_gen -p openai -m gpt-4o -n 3 -r 8 --evaluator_model gemini-1.5-flash-8b
  - Sokoban: python toybench_cli.py -t sokoban -p gemini -m gemini-2.0-flash -n 3 -r 40 --sokoban_level 0

Dev tips
- Add a task: create tasks/<name> prompts and environments/<name>_env.py; wire in get_environment and action parsing if special.
- Add a provider: implement LLMInterface subclass and register in toybench_cli.get_llm_interface.
- Evaluator provider is inferred from --evaluator_model prefix; can differ from agent provider.
- Solar requires <solar.html> tags; FS and Sokoban expect strict ```action``` blocks; invalid formats are treated as errors.
