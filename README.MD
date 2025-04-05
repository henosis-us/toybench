# ToyBench: Evaluating Agentic Planning, Action, and Iterative Refinement

## Vision & Goals

**ToyBench** evaluates the capability of Large Language Models (LLMs) to **plan, take actions agentically over time, and refine outputs based on feedback**. Our vision is to create a standardized method for measuring how well LLMs translate complex requirements and high-level goals into tangible, **"valid" sequences of actions or artifacts** within defined environments, assessed through a practical, multi-step process that may involve interaction and iteration.

Benchmark runs across **Tic-Tac-Toe** (strategic planning, unlimited turns), **File System** management (complex instruction following, state tracking, 50-round horizon), and iterative **Solar System** generation (visual refinement, code generation, multimodal feedback, 3-round horizon) demonstrate significant variation in capabilities between different models and highlight the distinct challenges posed by each task type and time horizon.

**Core Goals:**

1.  **Evaluate Agentic Capabilities:** Assess the LLM's ability to act as an agent – perceiving environmental state, planning, and executing actions or generating artifacts.
2.  **Measure Planning & Reasoning:** Test multi-step planning, reasoning, and feedback incorporation under task constraints.
3.  **Assess Action/Output Validity & Effectiveness:** Determine if actions/artifacts are valid and contribute effectively towards the goal over time.
4.  **Benchmark Goal Achievement:** Quantify success rates (`pass@1`, `pass@k`) across multiple attempts.
5.  **Analyze Performance Horizons / Refinement Cycles:** Understand how performance changes with allowed steps/rounds.
6.  **Compare Agent Performance:** Provide a consistent framework to compare LLMs/strategies across diverse agentic tasks.

## Core Concepts

*   **Agent:** The LLM being evaluated.
*   **Environment:** Simulated context (game, file system, browser loop) with rules, states, feedback.
*   **State:** Current situation provided to the agent.
*   **Goal:** High-level objective.
*   **Actions:** Permissible operations or generated artifacts.
*   **Turns/Steps/Rounds:** Interactions (State -> Action -> New State/Feedback). Limited by `--rounds` (except TTT).
*   **Validity:** Whether an action/artifact meets environment rules or basic structural requirements.
*   **Feedback (Iterative):** Information (logs, evaluations) about the previous action's outcome.
*   **Rendering (Visual):** Using tools (Selenium) to process output (HTML) into a representation (screenshot).
*   **Intermediate Evaluation (Iterative):** Assessment within a multi-step process for feedback (e.g., evaluating a screenshot).
*   **Final Evaluation:** Assessment at the end to determine the success score (1-3).

## Evaluation Methodology: Agent Interaction & Outcome Scoring

ToyBench uses interaction loops and outcome-based scoring:

1.  **Initialization:** Set up task environment and goal.
2.  **Interaction Loop (per Attempt, up to `--rounds` or game end):**
    *   Agent receives state/feedback + goal.
    *   Agent outputs action/artifact.
    *   Validation against rules/structure. Invalid output leads to failure/penalty.
    *   Valid action updates state (TTT, FS) or triggers rendering/intermediate eval (SolarGen).
    *   Environment generates new state/feedback.
3.  **Data Capture:** Log states, actions, validity, feedback, evaluations (`attempt_results.jsonl`).
4.  **Final Scoring (per Attempt):** Assess final state/artifact against criteria (`<task>_finaleval.txt` or deterministic checks).
    *   **Score 1 (Failure):** Goal not met, critical error, invalid action, failed eval.
    *   **Score 2 (Partial):** Progress made, valid outputs, but goal not fully met or minor errors.
    *   **Score 3 (Success):** Goal fully achieved via valid steps within limits, passing final eval.

## Reporting Metrics

*   **`pass@1` (Reliability):** `(# Successful Attempts) / (Total Attempts Completed)`. Average success rate per try.
*   **`pass@k` / `pass@20` (Capability):** 100% if *at least one* of the first `k` (typically 20) attempts succeeded, 0% otherwise. Often labeled "Any Success in Run".
*   **Overall ToyBench Score:** Can be calculated as the **Average `pass@1` score across all included tasks** *if run with a consistent round limit*. The results presented below use different effective horizons per task (TTT: Unlimited, FS: 50 MAX, SolarGen: 3), making a single overall score less directly comparable. Performance should be assessed considering the specific task constraints and relative strengths.

## Overall Performance Summary (Varied Horizons)

This table summarizes the performance of models tested across Tic-Tac-Toe (Unlimited Rounds), FileSystem, and SolarGen (3 Rounds). The "Overall Score" represents the average `pass@1` across these tasks. *Note: `gemini-2.0-flash-thinking-exp` is excluded from this overall summary due to missing FileSystem data.* Results are based on 20 attempts per task.

| Model                      | Overall Score (Avg. `pass@1`) | Notes                                                                     |
| :------------------------- | :---------------------------- | :------------------------------------------------------------------------ |
| `gemini-2.5-pro-exp-03-25` | **30.0%**                     | Strongest on TTT, capable on SolarGen, low `pass@1` but high partial on FS. |
| `gemini-2.0-flash`         | **28.3%**                     | Excellent on FS, but struggled significantly on TTT and SolarGen.         |
| `gemini-2.0-flash-lite`    | **25.0%**                     | Balanced moderate performance on TTT/FS, but failed SolarGen `pass@1`.    |
| `gemini-1.5-flash-8b`      | **0.0%**                      | Failed to achieve strict success (Score 3) on any task in these tests.    |

**Interpretation:**

*   The benchmark effectively differentiates model capabilities across diverse agentic challenges and horizons.
*   No single model excelled universally. `gemini-2.5-pro-exp` showed the best average performance, driven by strong strategic planning (TTT) and visual iteration capability (SolarGen), though its reliability on complex instruction following (FS `pass@1`) was lower than Flash likely due to the thinking step 2.5 pro engages in.
*   `gemini-2.0-flash` demonstrated exceptional proficiency in the highly structured `FileSystem` task but lacked the planning depth for TTT or the iterative refinement ability for `SolarGen`.
*   The 3-round horizon for SolarGen still proved challenging, with only `gemini-2.5-pro-exp` achieving any strict success (`pass@1`).

---

## Task-Specific Performance Details

Below are detailed results for each task, comparing all models tested under the specific round constraints reported in the logs. (20 attempts per model).

### Task: Tic-Tac-Toe (vs Optimal Opponent)

*   **Rounds:** Unlimited (Game Completion)
*   *Tests strategic planning, rule adherence, and goal achievement.*

| Model                             | `pass@1` (Win/Draw, No Invalid) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count |
| :-------------------------------- | :------------------------------ | :---------------------------- | :------------------------------ |
| `gemini-2.5-pro-exp-03-25`        | **65.0%**                       | 100%                          | 0                               |
| `gemini-2.0-flash-thinking-exp`\* | 35.0%                           | 100%                          | 4                               |
| `gemini-2.0-flash-lite`           | 10.0%                           | 100%                          | 0                               |
| `gemini-1.5-flash-8b`             | 0.0%                            | 0%                            | 0                               |
| `gemini-2.0-flash`                | 0.0%                            | 0%                            | 0                               |
*(`*` Model tested but excluded from overall average due to incomplete data across all tasks)*

**Commentary:** `gemini-2.5-pro-exp` clearly demonstrates superior strategic capability against an optimal opponent.

---

### Task: Complex File Organizer

*   **Rounds:**
*   *Tests complex instruction following, state tracking, conditional logic, command generation.*

| Model                             | `pass@1` (Perfect Match) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count |
| :-------------------------------- | :----------------------- | :---------------------------- | :------------------------------ |
| `gemini-2.0-flash`                | **85.0%**                | 100%                          | 2                               |
| `gemini-2.0-flash-lite`           | 65.0%                    | 100%                          | 1                               |
| `gemini-2.5-pro-exp-03-25`        | 15.0%                    | 100%                          | **17**                          |
| `gemini-1.5-flash-8b`             | 0.0%                     | 0%                            | 8                               |
| `gemini-2.0-flash-thinking-exp`\* | *N/A*                    | *N/A*                         | *N/A*                           |
*(`*` Data not available for this model due to rate limits)*

**Commentary:** `gemini-2.0-flash` excels at precise execution of complex, sequential instructions. `gemini-2.5-pro-exp` understands the goal but struggles with perfect execution (`pass@1`), achieving partial success frequently.

---

### Task: Solar System Generator (Iterative Refinement)

*  
*   *Tests HTML/JS code generation, incorporating visual feedback, debugging from logs.*

| Model                             | `pass@1` (>5 Planets + Moon) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count |
| :-------------------------------- | :--------------------------- | :---------------------------- | :------------------------------ |
| `gemini-2.5-pro-exp-03-25`        | **15.0%**                    | 80.0%                         | 13                              |
| `gemini-1.5-flash-8b`             | 0.0%                         | 10.0%                         | 2                               |
| `gemini-2.0-flash`                | 0.0%                         | 35.0%                         | 7                               |
| `gemini-2.0-flash-thinking-exp`\* | 0.0%                         | 25.0%                         | 5                               |
| `gemini-2.0-flash-lite`           | 0.0%                         | 20.0%                         | 4                               |
*(`*` Model tested but excluded from overall average due to incomplete data across all tasks)*

**Commentary:** Iterative tasks remain challenging. `gemini-2.5-pro-exp` is the only model demonstrating capability for full success, albeit unreliably. Most models achieve partial success (rendering *something* visible) but fail to meet the complex final criteria within 3 refinement rounds.

---

## Prerequisites & Usage

**(This section remains the same - includes Python version, dependencies, API keys, WebDriver setup, and example commands reflecting the different round settings)**

**Prerequisites:**

*   **Python:** Version 3.9+ recommended.
*   **Dependencies:** Install required packages:
    ```bash
    pip install google-generativeai python-dotenv requests selenium webdriver-manager
    ```
*   **API Keys:** Set `GOOGLE_API_KEY` in environment or `.env` file.
*   **WebDriver (for `solar_gen`):** Install Chrome/Chromium and ensure `chromedriver` (matching version) is in PATH or managed by `webdriver-manager`.

**Usage Examples:**

```bash
# --- Run TicTacToe (Unlimited Rounds) ---
python toybench_cli.py --task tic_tac_toe --provider gemini --model gemini-2.5-pro-exp-03-25 --attempts 20

# --- Run Complex File System ---
# Use --rounds 50 or similar for a high cap if desired
python toybench_cli.py --task file_system --provider gemini --model gemini-2.0-flash --attempts 20 --rounds 50

# --- Run Solar System Generator ---
python toybench_cli.py --task solar_gen --provider gemini --model gemini-2.5-pro-exp-03-25 --attempts 20 --rounds 3

# --- Run Solar System Generator (Different Horizon - e.g., 3 Rounds) ---
python toybench_cli.py --task solar_gen --provider gemini --model gemini-2.5-pro-exp-03-25 --attempts 20 --rounds 3

# --- Debug Run ---
python toybench_cli.py --task solar_gen --attempts 1 --rounds 3 --log_level DEBUG