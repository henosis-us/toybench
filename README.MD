# ToyBench: Evaluating Agentic Planning, Action, and Iterative Refinement

**ToyBench** evaluates the capability of Large Language Models (LLMs) to **plan, take actions agentically over time, and refine outputs based on feedback**. Our vision is to create a method for measuring how well LLMs translate complex requirements and high-level goals into tangible, **"valid" sequences of actions or artifacts** within defined environments, assessed through a practical, multi-step process that may involve interaction and iteration. This benchmark will be continually updated over time.

Benchmark runs across **Tic-Tac-Toe** (strategic planning, multi-turn horizon), **File System** management (complex instruction following, state tracking, high-round horizon), and iterative **Solar System** generation (visual refinement, code generation, multimodal feedback, limited-round horizon) demonstrate significant variation in capabilities between different models and providers and highlight the distinct challenges posed by each task type and time horizon.

New results for `gemini-2.5-pro-preview-06-05` show a massive leap in performance and efficiency, securing the #2 spot overall. Claude Opus remains the top performer, excelling in the most complex agentic tasks. For the Solar System task, a **Differentiated Score** has been introduced to better capture partial successes, calculated as: ((Number of Score 3 attempts * 1.0) + (Number of Score 2 attempts * 0.3)) / Total Attempts * 100.

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

For the Solar System Generator task, a **Differentiated Score** is now used in reporting to better reflect partial achievements. This score is calculated as ((Number of Score 3 attempts * 1.0) + (Number of Score 2 attempts * 0.3)) / Total Attempts * 100, providing a weighted average that highlights models capable of iterative refinement even if not fully successful.

## Reporting Metrics

*   **`pass@1` (Reliability):** `(# Successful Attempts) / (Total Attempts Completed)`. Average success rate per try.
*   **`pass@k` / `pass@20` (Capability):** 100% if *at least one* of the first `k` (typically 20) attempts succeeded (Score 3), 0% otherwise. Often labeled "Any Success in Run".
*   **Differentiated Score (SolarGen-Specific):** For the Solar System task, a weighted score incorporating partial successes (Score 2 at 0.3 weight) to better differentiate models.
*   **Overall ToyBench Score:** Calculated as the **Average of TTT `pass@1`, FileSystem `pass@1`, and SolarGen Differentiated Score across all included tasks** if run with a consistent attempt count. The results presented below use 20 attempts per task where data is available. Models are only included if they have a completed run (20 attempts) for all three tasks (Tic-Tac-Toe, FileSystem, SolarGen).

## Overall Performance Summary (Varied Horizons, 20 Attempts per Task)

This table summarizes the performance of models tested across Tic-Tac-Toe, FileSystem, and SolarGen. The "Overall Score" represents the average of TTT `pass@1`, FileSystem `pass@1`, and SolarGen Differentiated Score based on available 20/20 runs for all three tasks.

| Model                                              | Provider        | Overall Score (Avg. of Scores) | Notes                                                                                                                                                                                                                           |
| :------------------------------------------------- | :-------------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| claude-opus-4                                      | anthropic       | **84.67%**                     | Strong on all tasks; Solar Differentiated Score: 79.00%. Estimated cost ~$48.00 per full run.                                                                                                    |
| gemini-2.5-pro-preview-06-05                       | gemini          | **65.00%**                     | Jumps to #2. Perfect on TTT (100%), much improved on FS (50%). Solar Differentiated Score: 45.00%. Highly efficient, cost ~$11.43 per run.                                                                  |
| o3                                                 | openai          | **56.83%**                     | Strong overall; high estimated cost (~$26.27 per full run). Solar Differentiated Score: 30.50%.                                                                                                                                |
| grok-3-mini-beta-B8                                | quality_compute | **55.50%**                     | Best of 8 variant; (~$2.06 per full run). Solar Differentiated Score: 21.50%.                                                                                                                                        |
| claude-sonnet-4                                    | anthropic       | **54.50%**                     | Strong on Solar Gen; Solar Differentiated Score: 68.50%. Estimated cost ~$10.05 per full run.                                                                                                       |
| gemini-2.5-flash-preview-05-20                     | gemini          | **54.50%**                     | Solar Differentiated Score: 13.50%. Improved significantly from previous flash. Estimated cost ~$4.02.                                                                                                          |
| grok-3-mini-beta                                   | grok            | **40.50%**                     | Excellent on TTT (90.00%), but low on FS and Solar. Solar Differentiated Score: 16.50%. Cost-effective (~$0.373 per full run).                                                                                                 |
| gemini-2.5-pro-exp-03-25                           | gemini          | **38.17%**                     | Good on TTT and Solar. Solar Differentiated Score: 34.50%. March release.                                                                                                                                       |
| gpt-4.1                                            | openai          | **33.83%**                     | Excellent on FileSystem (100.00%), but weak on others. Solar Differentiated Score: 1.50%.                                                                                                                         |
| o4-mini                                            | openai          | **32.50%**                     | Good on FS (70.00%), weak on TTT and Solar. Solar Differentiated Score: 7.50%.                                                                                                                                    |
| gemini-2.5-flash-lite-preview-06-17 16k thinking   | gemini          | **32.33%**                     | Thinking mode variant. Strong on TTT (75%). Estimated cost ~$0.34 per full run.                                                                                                                                 |
| gemini-2.5-pro-preview-05-06                       | gemini          | **31.00%**                     | Solar Differentiated Score: 43.00%. Cost corrected to ~$24.73 based on re-analysis of token logs with "thinking" billed at output rate.                                                                                 |
| gpt-4.1-mini                                       | openai          | **28.33%**                     | Strong on FS (85.00%), but failed on TTT and Solar. Solar Differentiated Score: 0.00%.                                                                                                                            |
| deepseek-chat                                      | deepseek        | **18.00%**                     | Moderate on TTT and FS, failed on Solar. Solar Differentiated Score: 9.00%. Provider shown as base DeepSeek.                                                                                                   |
| deepseek-reasoner                                  | deepseek        | **10.50%**                     | Some success on Solar Differentiated Score (21.50%), weak on TTT and FS. Provider shown as base DeepSeek.                                                                                                     |
| gemini-2.5-flash-lite-preview-06-17                | gemini          | **4.67%**                      | Standard non-thinking version. Low scores on all tasks. Estimated cost ~$0.30 per full run.                                                                                                                     |
| gpt-4.1-nano                                       | openai          | **1.67%**                      | Minimal success. Solar Differentiated Score: 0.00%.                                                                                                                                                               |
| gemini-1.5-flash-8b                                | gemini          | **0.00%**                      | Failed strict success on any task. Solar Differentiated Score: 0.00%.                                                                                                                                           |

**Notes:**
- Costs for Gemini Pro models have been updated based on detailed log analysis, billing "thinking" tokens at the output rate. This significantly increased the effective cost of the inefficient `05-06` model, highlighting the value of the new `06-05` version's efficiency.

## Task-Specific Performance Details

Below are detailed results for each task, comparing all models tested under the specific round constraints reported in the logs, filtering for runs with 20/20 attempts unless noted otherwise. Tables have been updated with new data.

### Task: Tic-Tac-Toe (vs Optimal Opponent)
*   **Rounds:** Typically 5, 8, or 10 turns max.
*   *Tests strategic planning, rule adherence, and goal achievement.*

| Model                                            | Provider        | Turn Horizon | Attempts Run | `pass@1` (Win/Draw, No Invalid) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes                         |
| :----------------------------------------------- | :-------------- | :----------- | :----------- | :------------------------------ | :---------------------------- | :------------------------------ | :--------------------- | :------------------- | :---------------------------- |
| gemini-2.5-pro-preview-06-05                     | gemini          | 9            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    | Perfect score.                |
| claude-sonnet-4 16k thinking                     | anthropic       | 8            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    | New variant; perfect performance. |
| grok-3-mini-beta-B8                              | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    |                               |
| grok-3-mini-beta                                 | grok            | 5            | 20 / 20      | 90.00%                          | 100%                          | 2                               | 0                      | 0                    |                               |
| o3                                               | openai          | 5            | 20 / 20      | 85.00%                          | 100%                          | 1                               | 2                      | 0                    | Corrected grading applied.    |
| claude-opus-4                                    | anthropic       | 8            | 20 / 20      | 80.00%                          | 100%                          | 4                               | 0                      | 0                    | New data; strong performance. |
| gemini-2.5-flash-preview-05-20                   | gemini          | 8            | 20 / 20      | 80.00%                          | 100%                          | 4                               | 0                      | 0                    | Significant jump.             |
| claude-sonnet-4                                  | anthropic       | 8            | 20 / 20      | 75.00%                          | 100%                          | 5                               | 0                      | 0                    |                               |
| gemini-2.5-flash-lite-preview-06-17 16k thinking | gemini          | 9            | 20 / 20      | 75.00%                          | 100%                          | 2                               | 0                      | 3                    | Thinking mode variant.        |
| gemini-2.5-pro-exp-03-25                         | gemini          | 10           | 20 / 20      | 65.00%                          | 100%                          | 0                               | 7                      | 0                    | March release                 |
| gemini-2.0-flash-thinking-exp-1219               | gemini          | 10           | 20 / 20      | 35.00%                          | 100%                          | 4                               | 8                      | 1                    |                               |
| gemini-2.5-pro-preview-05-06                     | gemini          | 5            | 20 / 20      | 30.00%                          | 100%                          | 13                              | 1                      | 0                    | Based on combined reports.    |
| o4-mini                                          | openai          | 5            | 20 / 20      | 20.00%                          | 100%                          | 13                              | 3                      | 0                    |                               |
| gemini-2.5-flash-lite-preview-06-17              | gemini          | 9            | 20 / 20      | 5.00%                           | 100%                          | 18                              | 0                      | 1                    | Standard version.             |
| gpt-4.1-nano                                     | openai          | 5            | 20 / 20      | 5.00%                           | 100%                          | 0                               | 19                     | 0                    |                               |
| gpt-4.1-mini                                     | openai          | 5            | 20 / 20      | 0.00%                           | 0%                            | 3                               | 13                     | 4                    |                               |
| gpt-4.1                                          | openai          | 30           | 20 / 20      | 0.00%                           | 0%                            | 0                               | 20                     | 0                    |                               |

**Commentary:** Tic-Tac-Toe remains a strong differentiator. The `gemini-2.5-pro-preview-06-05` achieves a perfect score, demonstrating mastery of short-term strategic planning and a massive improvement in token efficiency over its predecessor.

### Task: Complex File Organizer
*   **Rounds:** Typically 10 or 35 turns max.
*   *Tests complex instruction following, state tracking, conditional logic, command generation.*

| Model                                            | Provider        | Turn Horizon | Attempts Run | `pass@1` (Perfect Match) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes                              |
| :----------------------------------------------- | :-------------- | :----------- | :----------- | :----------------------- | :---------------------------- | :------------------------------ | :--------------------- | :------------------- | :--------------------------------- |
| gpt-4.1                                          | openai          | 30           | 20 / 20      | 100.00%                  | 100%                          | 0                               | 0                      | 0                    |                                    |
| claude-opus-4                                    | anthropic       | 35           | 20 / 20      | 95.00%                   | 100%                          | 1                               | 0                      | 4                    | New data; high success rate.       |
| gpt-4.1-mini                                     | openai          | 30           | 20 / 20      | 85.00%                   | 100%                          | 2                               | 1                      | 0                    |                                    |
| gemini-2.0-flash                                 | gemini          | 35           | 20 / 20      | 85.00%                   | 100%                          | 2                               | 0                      | 1                    |                                    |
| gemini-2.5-flash-preview-05-20                   | gemini          | 35           | 20 / 20      | 70.00%                   | 100%                          | 6                               | 0                      | 0                    | Improvement.                       |
| o4-mini                                          | openai          | 30           | 20 / 20      | 70.00%                   | 100%                          | 6                               | 0                      | 0                    |                                    |
| o3                                               | openai          | 30           | 20 / 20      | 55.00%                   | 100%                          | 8                               | 1                      | 0                    |                                    |
| gemini-2.5-pro-preview-06-05                     | gemini          | 35           | 20 / 20      | 50.00%                   | 100%                          | 10                              | 0                      | 0                    | Major improvement.                 |
| grok-3-mini-beta-B8                              | quality_compute | 35           | 20 / 20      | 45.00%                   | 100%                          | 11                              | 0                      | 0                    |                                    |
| claude-sonnet-4 16k thinking                     | anthropic       | 35           | 20 / 20      | 20.00%                   | 100%                          | 16                              | 0                      | 0                    | New variant; high partial success. |
| claude-sonnet-4                                  | anthropic       | 35           | 20 / 20      | 20.00%                   | 100%                          | 16                              | 0                      | 0                    |                                    |
| deepseek-chat                                    | deepseek        | 35           | 20 / 20      | 20.00%                   | 100%                          | 9                               | 7                      | 0                    |                                    |
| gemini-2.5-pro-preview-05-06                     | gemini          | 30           | 20 / 20      | 20.00%                   | 100%                          | 16                              | 0                      | 0                    | New data (May I/O); Improvement.   |
| gemini-2.5-pro-exp-03-25                         | gemini          | 35           | 20 / 20      | 15.00%                   | 100%                          | 17                              | 0                      | 0                    | March release                      |
| grok-3-mini-beta                                 | grok            | 35           | 20 / 20      | 15.00%                   | 100%                          | 12                              | 5                      | 0                    |                                    |
| gemini-2.5-flash-lite-preview-06-17 16k thinking | gemini          | 35           | 20 / 20      | 10.00%                   | 100%                          | 10                              | 7                      | 1                    | Thinking mode variant.             |
| gemini-2.5-flash-lite-preview-06-17              | gemini          | 35           | 20 / 20      | 0.00%                    | 0%                            | 11                              | 3                      | 6                    | Standard version.                  |

**Commentary:** File System tests precise execution over a long time horizon. Claude Opus-4 leads among high-capability models. The `gemini-2.5-pro-preview-06-05` shows a dramatic improvement, jumping from 20% to 50% success, indicating much stronger state tracking and instruction following capabilities.

### Task: Solar System Generator (Iterative Refinement)
*   **Rounds:** Typically 3 turns max.
*   *Tests HTML/JS code generation, incorporating visual feedback, debugging from logs. Now includes a Differentiated Score for better nuance, weighting partial successes at 0.3.*

| Model                                            | Provider        | Turn Horizon | Attempts Run | `pass@1` (>5 Planets + Moon) | Differentiated Score (S3=1pt, S2=0.3pt) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes                                |
| :----------------------------------------------- | :-------------- | :----------- | :----------- | :--------------------------- | :------------------------------------- | :------------------------------ | :--------------------- | :------------------- | :----------------------------------- |
| claude-opus-4                                    | anthropic       | 3            | 20 / 20      | 70.00%                       | 79.00%                                 | 5                               | 4                      | 1                    | Strong iterative refinement.         |
| claude-sonnet-4                                  | anthropic       | 3            | 20 / 20      | 55.00%                       | 68.50%                                 | 9                               | 0                      | 0                    | High partial success rate.           |
| gemini-2.5-pro-preview-06-05                     | gemini          | 3            | 20 / 20      | 30.00%                       | 45.00%                                 | 9                               | 5                      | 0                    | Continued progress.                  |
| gemini-2.5-pro-preview-05-06                     | gemini          | 3            | 20 / 20      | 22.50%                       | 43.00%                                 | 12                              | 2                      | 0                    | Based on combined reports.           |
| gemini-2.5-pro-exp-03-25                         | gemini          | 3            | 20 / 20      | 15.00%                       | 34.50%                                 | 13                              | 0                      | 0                    | March release.                       |
| o3                                               | openai          | 3            | 20 / 20      | 20.00%                       | 30.50%                                 | 7                               | 9                      | 0                    | Previous SOTA on pass@1.             |
| deepseek-reasoner-B1                             | deepseek        | 3            | 20 / 20      | 5.00%                        | 21.50%                                 | 11                              | 8                      | 0                    |                                    |
| grok-3-mini-beta-B8                              | quality_compute | 3            | 20 / 20      | 5.00%                        | 21.50%                                 | 11                              | 8                      | 0                    |                                    |
| gemini-2.5-flash-preview-05-20                   | gemini          | 3            | 20 / 20      | 0.00%                        | 13.50%                                 | 9                               | 11                     | 0                    |                                    |
| gemini-2.5-flash-lite-preview-06-17 16k thinking | gemini          | 3            | 20 / 20      | 0.00%                        | 12.00%                                 | 8                               | 12                     | 0                    | Thinking mode variant.               |
| gemini-2.5-flash-lite-preview-06-17              | gemini          | 3            | 20 / 20      | 0.00%                        | 9.00%                                  | 6                               | 14                     | 0                    | Standard version.                    |
| deepseek-chat                                    | deepseek        | 3            | 20 / 20      | 0.00%                        | 9.00%                                  | 6                               | 14                     | 0                    |                                    |
| gpt-4.1                                          | openai          | 3            | 20 / 20      | 0.00%                        | 1.50%                                  | 1                               | 19                     | 0                    |                                    |

**Commentary:** Iterative refinement remains a major hurdle. Claude Opus-4 still leads with a high differentiated score. The `gemini-2.5-pro-preview-06-05` continues the upward trend for Gemini models, posting solid gains in both `pass@1` and the Differentiated Score, showing improved visual and code-based iteration.

## Prerequisites & Usage

**Prerequisites:**
*   **Python:** Version 3.9+ recommended.
*   **Dependencies:** Install required packages:
    ```bash
    pip install google-generativeai python-dotenv requests selenium webdriver-manager
    ```
*   **API Keys:** Set `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `QUALITY_COMPUTE_API_KEY` in environment or `.env` file as needed for the providers you wish to test.
*   **WebDriver (for `solar_gen`):** Install Chrome/Chromium and ensure `chromedriver` (matching version) is in PATH or managed by `webdriver-manager`.

**Usage Examples:**
```bash
# --- Run TicTacToe ---
# Use --rounds to set the max turn horizon if desired (e.g., --rounds 5)
python toybench_cli.py --task tic_tac_toe --provider grok --model grok-3-mini-beta --attempts 20 --rounds 5

# --- Run Complex File System ---
# Use --rounds 50 or similar for a high cap if desired (e.g., --rounds 35 or --rounds 50)
python toybench_cli.py --task file_system --provider openai --model gpt-4.1 --attempts 20 --rounds 30

# --- Run Solar System Generator ---
# Use --rounds 3 or similar for a limited horizon
python toybench_cli.py --task solar_gen --provider gemini --model gemini-2.5-pro-preview-06-05 --attempts 20 --rounds 3

# --- Run with a Best of N variant (example using quality_compute) ---
python toybench_cli.py --task tic_tac_toe --provider quality_compute --model grok-3-mini-beta-B8 --attempts 20 --rounds 5

# --- Debug Run ---
python toybench_cli.py --task solar_gen --attempts 1 --rounds 3 --log_level DEBUG