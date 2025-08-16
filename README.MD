# ToyBench: Evaluating Agentic Planning, Action, and Iterative Refinement
**ToyBench** evaluates the capability of Large Language Models (LLMs) to **plan, take actions agentically over time, and refine outputs based on feedback**. Our vision is to create a method for measuring how well LLMs translate complex requirements and high-level goals into tangible, **"valid" sequences of actions or artifacts** within defined environments, assessed through a practical, multi-step process that may involve interaction and iteration. This benchmark will be continually updated over time.

Benchmark runs across **Tic-Tac-Toe** (strategic planning, multi-turn horizon), **File System** management (complex instruction following, state tracking, high-round horizon), and iterative **Solar System** generation (visual refinement, code generation, multimodal feedback, limited-round horizon) demonstrate significant variation in capabilities between different models and providers.

This update introduces results for the **GPT-5 family**, with **`gpt-5 (high)` taking the #1 spot overall**. Its performance, particularly on the File System task, suggests it overcomes weaknesses observed in other "thinking-class" models, behaving more like a true agent capable of robust, long-horizon execution. For the Solar System task, a **Differentiated Score** is used to better capture partial successes, calculated as: `((Score 3 attempts * 1.0) + (Score 2 attempts * 0.3)) / Total Attempts * 100`.

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
*   **Turns/Steps/Rounds:** Interactions (State -> Action -> New State/Feedback). Limited by `--rounds`.
*   **Validity:** Whether an action/artifact meets environment rules or basic structural requirements.
*   **Feedback (Iterative):** Information (logs, evaluations) about the previous action's outcome.
*   **Rendering (Visual):** Using tools (Selenium) to process output (HTML) into a representation (screenshot).
*   **Intermediate Evaluation (Iterative):** Assessment within a multi-step process for feedback.
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
4.  **Final Scoring (per Attempt):** Assess final state/artifact against criteria.
    *   **Score 1 (Failure):** Goal not met, critical error, invalid action, failed eval.
    *   **Score 2 (Partial):** Progress made, valid outputs, but goal not fully met or minor errors.
    *   **Score 3 (Success):** Goal fully achieved via valid steps within limits, passing final eval.

For the Solar System Generator task, a **Differentiated Score** is now used in reporting to better reflect partial achievements, providing a weighted average that highlights models capable of iterative refinement even if not fully successful.

## Reporting Metrics
*   **`pass@1` (Reliability):** `(# Successful Attempts) / (Total Attempts Completed)`.
*   **`pass@k` / `pass@20` (Capability):** 100% if *at least one* of the first `k` (typically 20) attempts succeeded (Score 3), 0% otherwise.
*   **Differentiated Score (SolarGen-Specific):** A weighted score incorporating partial successes (Score 2 at 0.3 weight).
*   **Overall ToyBench Score:** Calculated as the **Average of TTT `pass@1`, FileSystem `pass@1`, and SolarGen Differentiated Score**. The results below use 20 attempts per task where data is available. Models are only included if they have completed runs for all three tasks.

## Overall Performance Summary (Standardized to 20 Attempts per Task)
This table summarizes the performance of models tested across Tic-Tac-Toe, FileSystem, and SolarGen. The "Overall Score" is the average of the three task scores.

| Model | Provider | Overall Score | Notes |
| :--- | :--- | :--- | :--- |
| **gpt-5 (high)** | openai | **93.67%** | **New #1**. Perfect on TTT & FS (100%), strong on Solar (81% DSS). Estimated cost ~$18.77. |
| claude-opus-4 | anthropic | **84.67%** | Strong on all tasks; FS Pass@1: 95%, Solar DSS: 79.00%. Estimated cost ~$48.00. |
| gemini-2.5-pro-preview-06-05 | gemini | **65.00%** | Perfect on TTT (100%), improved on FS (50%). Solar DSS: 45.00%. Highly efficient, cost ~$11.43. |
| o3 | openai | **56.83%** | Strong overall; high estimated cost (~$5.25 with corrected pricing). Solar DSS: 30.50%. |
| grok-3-mini-beta-B8 | quality_compute | **55.50%** | Best of 8 variant; (~$2.06 per full run). Solar DSS: 21.50%. |
| claude-sonnet-4 | anthropic | **54.50%** | Strong on Solar Gen; Solar DSS: 68.50%. Estimated cost ~$10.05. |
| gemini-2.5-flash-preview-05-20 | gemini | **54.50%** | Solar DSS: 13.50%. Improved from previous flash. Estimated cost ~$4.02. |
| **gpt-5-mini** | openai | **53.83%** | **Updated:** Strong FS (90%), improved TTT (50%). Solar DSS: 21.50%. Estimated cost ~$2.60. |
| **gpt-5-nano** | openai | **49.63%** | Strong TTT (80%), mid-tier FS (58%). Solar DSS: 11.00%. Estimated cost ~$1.14. |
| kimi-k2-0711-preview | kimi | **45.26%** | Solar Differentiated Score: 20.00%. Estimated cost ~$1.49. |
| grok-4 | grok | **42.80%** | Perfect TTT (100%), but weak on other tasks. Solar DSS: 28.40%. Estimated cost ~$23.10. |
| grok-3-mini-beta | grok | **40.50%** | Excellent on TTT (90.00%), but low on FS and Solar. Solar DSS: 16.50%. Cost-effective (~$0.373). |
| gpt-4.1 | openai | **33.83%** | Excellent on FileSystem (100.00%), but weak on others. Solar DSS: 1.50%. |

**Notes:**
- Costs are estimated for a full 60-attempt run (20 per task) based on token logs and stated pricing.
- `gpt-5 (high)` File System score is based on reruns that corrected for an environment-specific `ls` behavior.
- `gpt-5-mini`'s Tic-Tac-Toe score has been updated to 50% after a re-run, resolving a previous performance anomaly.

## Task-Specific Performance Details
Below are detailed results for each task. Tables are re-sorted with the latest data.

### Task: Tic-Tac-Toe (vs Optimal Opponent)
*   **Rounds:** Varies by run.
*   *Tests strategic planning, rule adherence, and goal achievement.*

| Model | Provider | `pass@1` | Partial (S2) | Failed (S1) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **gpt-5 (high)** | openai | **100.00%** | 0 | 0 | Perfect score. |
| gemini-2.5-pro-preview-06-05 | gemini | **100.00%** | 0 | 0 | Perfect score. |
| grok-4 | grok | **100.00%** | 0 | 0 | Perfect score. |
| grok-3-mini-beta-B8 | quality_compute | **100.00%** | 0 | 0 | Best of 8 variant. |
| grok-3-mini-beta | grok | **90.00%** | 2 | 0 | |
| o3 | openai | **85.00%** | 1 | 2 | |
| **gpt-5-nano** | openai | **80.00%** | 4 | 0 | Stronger than Mini. |
| claude-opus-4 | anthropic | **80.00%** | 4 | 0 | |
| claude-sonnet-4 | anthropic | **75.00%** | 5 | 0 | |
| gemini-2.5-pro-exp-03-25 | gemini | **65.00%** | 0 | 7 | |
| **gpt-5-mini** | openai | **50.00%** | 10 | 0 | **Updated:** Re-run shows improved performance. |
| gpt-4.1 | openai | **0.00%** | 0 | 20 | |

**Commentary:** Tic-Tac-Toe remains a strong differentiator. `gpt-5 (high)` joins other top models with a perfect score. `gpt-5-mini`'s re-run resolves a previous anomaly, showing mid-tier performance.

### Task: Complex File Organizer
*   **Rounds:** Typically 35 turns max.
*   *Tests complex instruction following, state tracking, and long-horizon execution.*

| Model | Provider | `pass@1` | Partial (S2) | Failed (S1) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **gpt-5 (high)** | openai | **100.00%** | 0 | 0 | Perfect score after re-evaluation. |
| gpt-4.1 | openai | **100.00%** | 0 | 0 | |
| claude-opus-4 | anthropic | **95.00%** | 1 | 0 | |
| **gpt-5-mini** | openai | **90.00%** | 2 | 0 | Very strong performance. |
| gpt-4.1-mini | openai | **85.00%** | 2 | 1 | |
| **gpt-5-nano** | openai | **57.89%** | 7 | 1 | Based on 19 non-error attempts. |
| o3 | openai | **55.00%** | 8 | 1 | |
| gemini-2.5-pro-preview-06-05 | gemini | **50.00%** | 10 | 0 | Major improvement. |
| grok-3-mini-beta-B8 | quality_compute | **45.00%** | 11 | 0 | |

**Commentary:** `gpt-5 (high)` demonstrates mastery of long-horizon execution, a task where many "thinking-mode" models struggle. Its ability to maintain state and execute precise tool commands over dozens of steps suggests it has overcome common agentic failure modes.

### Task: Solar System Generator (Iterative Refinement)
*   **Rounds:** 3 turns max.
*   *Tests HTML/JS code generation and incorporating visual feedback. Scored with DSS.*

| Model | Provider | Differentiated Score | `pass@1` (S3) | Partial (S2) | Failed (S1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **gpt-5 (high)** | openai | **81.00%** | 75.00% | 4 | 1 |
| claude-opus-4 | anthropic | **79.00%** | 70.00% | 6 | 0 |
| claude-sonnet-4 | anthropic | **68.50%** | 55.00% | 9 | 0 |
| gemini-2.5-pro-preview-06-05 | gemini | **45.00%** | 30.00% | 9 | 5 |
| **gpt-5-mini** | openai | **21.50%** | 20.00% | 1 | 15 |
| deepseek-reasoner-B1 | deepseek | **21.50%** | 5.00% | 11 | 8 |
| grok-3-mini-beta-B8 | quality_compute | **21.50%** | 5.00% | 11 | 8 |
| **gpt-5-nano** | openai | **11.00%** | 5.00% | 4 | 15 |
| gpt-4.1 | openai | **1.50%** | 0.00% | 1 | 19 |

**Commentary:** Iterative refinement is a major hurdle. `gpt-5 (high)` shows best-in-class performance, consistently turning feedback into valid code improvements. The Mini and Nano variants struggle significantly with this creative and corrective task.

## Prerequisites & Usage
**Prerequisites:**
*   **Python:** Version 3.9+ recommended.
*   **Dependencies:** Install required packages:
    ```bash
    pip install google-generativeai python-dotenv requests selenium webdriver-manager
    ```
*   **API Keys:** Set `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, etc., in environment or `.env` file.
*   **WebDriver (for `solar_gen`):** Install Chrome/Chromium and ensure `chromedriver` is in PATH.

**Usage Examples:**
```bash
# Run TicTacToe for GPT-5 (high)
python toybench_cli.py --task tic_tac_toe --provider openai --model gpt-5 --attempts 20 --rounds 10

# Run File System for GPT-5 Mini
python toybench_cli.py --task file_system --provider openai --model gpt-5-mini --attempts 20 --rounds 35

# Run Solar System for Claude Opus
python toybench_cli.py --task solar_gen --provider anthropic --model claude-opus-4-20250514 --attempts 20 --rounds 3