# ToyBench: Evaluating Agentic Planning, Action, and Iterative Refinement

**ToyBench** evaluates the capability of Large Language Models (LLMs) to **plan, take actions agentically over time, and refine outputs based on feedback**. Our vision is to create a method for measuring how well LLMs translate complex requirements and high-level goals into tangible, **"valid" sequences of actions or artifacts** within defined environments, assessed through a practical, multi-step process that may involve interaction and iteration. This benchmark will be continually updated over time.

Benchmark runs across **Tic-Tac-Toe** (strategic planning, multi-turn horizon), **File System** management (complex instruction following, state tracking, high-round horizon), and iterative **Solar System** generation (visual refinement, code generation, multimodal feedback, limited-round horizon) demonstrate significant variation in capabilities between different models and providers and highlight the distinct challenges posed by each task type and time horizon. 

For the Solar System task, a new **Differentiated Score** has been introduced to better capture partial successes, calculated as: ((Number of Score 3 attempts * 1.0) + (Number of Score 2 attempts * 0.3)) / Total Attempts * 100. This provides a more nuanced evaluation, accounting for models that make progress but do not fully achieve the goal.

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
*   **Overall ToyBench Score:** Calculated as the **Average of TTT `pass@1`, FileSystem `pass@1`, and SolarGen Differentiated Score across all included tasks** if run with a consistent attempt count. The results presented below use 20 attempts per task where data is available. Models are only included if they have a completed run (20 attempts) for all three tasks (Tic-Tac-Toe, FileSystem, SolarGen). Grok "fast" variants and "-B1" QualityCompute variants are grouped with their base models for this summary where appropriate or if the base model was tested under that provider.

## Overall Performance Summary (Varied Horizons, 20 Attempts per Task)
This table summarizes the performance of models tested across Tic-Tac-Toe, FileSystem, and SolarGen. The "Overall Score" represents the average of TTT `pass@1`, FileSystem `pass@1`, and SolarGen Differentiated Score based on available 20/20 runs for all three tasks.

| Model                               | Provider        | Overall Score (Avg. of Scores) | Notes                                                                                                                                                                                                                           |
| :---------------------------------- | :-------------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| o3                                  | openai          | **56.83%**                     | Strong overall; high estimated cost (~$26.27 per full run). Solar Differentiated Score: 30.50%.                                                                                                                                |
| grok-3-mini-beta-B8                 | quality_compute | **55.50%**                     | Best of 8 variant; high cost (~$2.06 per full run). Solar Differentiated Score: 21.50%.                                                                                                                                        |
| gemini-2.5-flash-preview-05-20      | gemini          | **54.50%**                     | **NEW FLASH SOTA!** Solar Differentiated Score: 13.50%. Improved significantly from previous flash. Estimated cost ~$4.02.                                                                                                     |
| grok-3-mini-beta                    | grok            | **40.50%**                     | Excellent on TTT (90.00%), but low on FS and Solar. Solar Differentiated Score: 16.50%. Cost-effective (~$0.373 per full run).                                                                                                 |
| gemini-2.5-pro-exp-03-25            | gemini          | **38.17%**                     | Good on TTT and Solar. Solar Differentiated Score: 34.50%. March release.                                                                                                 |
| gpt-4.1                             | openai          | **33.83%**                     | Excellent on FileSystem (100.00%), but weak on others. Solar Differentiated Score: 1.50%.                                                                                  |
| o4-mini                             | openai          | **32.50%**                     | Good on FS (70.00%), weak on TTT and Solar. Solar Differentiated Score: 7.50%.                                                                                            |
| gpt-4.1-mini                        | openai          | **28.33%**                     | Strong on FS (85.00%), but failed on TTT and Solar. Solar Differentiated Score: 0.00%.                                                                                    |
| gemini-2.5-pro-preview-05-06        | gemini          | **31.00%**                     | New SOTA on Solar Differentiated Score (43.00%)! Improved on FS but regressed on TTT. Overall slight improvement. Estimated cost ~$16.58. (May I/O release).               |
| deepseek-chat                       | deepseek        | **18.00%**                     | Moderate on TTT and FS, failed on Solar. Solar Differentiated Score: 9.00%. Provider shown as base DeepSeek.                                                              |
| deepseek-reasoner                   | deepseek        | **10.50%**                     | Some success on Solar Differentiated Score (21.50%), weak on TTT and FS. Provider shown as base DeepSeek.                                                                |
| gpt-4.1-nano                        | openai          | **1.67%**                      | Minimal success. Solar Differentiated Score: 0.00%.                                                                                                                       |
| gemini-1.5-flash-8b                 | gemini          | **0.00%**                      | Failed strict success on any task. Solar Differentiated Score: 0.00%.                                                                                                     |

**Notes:**
*   The benchmark continues to differentiate model capabilities. The introduction of the SolarGen Differentiated Score (weighting Score 2 at 0.3) provides a more nuanced evaluation, highlighting models that make partial progress. This update benefits models like `gemini-2.5-pro-exp-03-25` and `grok-3-beta`, which have higher partial success rates.
*   `o3` remains the top performer overall. The new Gemini Flash model (`05-20`) achieves a new flash state-of-the-art overall score of 54.50%, driven by improvements in TTT and FS, with its Solar Differentiated Score adding nuance.
*   `grok-3-mini-beta` and its `quality_compute` variants excel at Tic-Tac-Toe
*   `gpt-4.1` leads on the FileSystem task with 100% `pass@1`.
*   SolarGen remains challenging, but clearly diffentiates even models within the same family, with `gemini-2.5-pro-preview-05-06` at 43.00% and `gemini-2.5-pro-exp-03-25` at 34.50%.

## Task-Specific Performance Details
Below are detailed results for each task, comparing all models tested under the specific round constraints reported in the logs, filtering for runs with 20/20 attempts unless noted otherwise. Tables have been updated with new data from the provided reports.

### Task: Tic-Tac-Toe (vs Optimal Opponent)
*   **Rounds:** Typically 5 or 10 turns max (implicitly unlimited until game ends in some runs).
*   *Tests strategic planning, rule adherence, and goal achievement.*

| Model                                    | Provider        | Turn Horizon | Attempts Run | `pass@1` (Win/Draw, No Invalid) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes                                  |
| :--------------------------------------- | :-------------- | :----------- | :----------- | :------------------------------ | :---------------------------- | :------------------------------ | :--------------------- | :------------------- | :------------------------------------- |
| quality_compute/grok-3-mini-beta-B128    | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    |                                        |
| quality_compute/grok-3-mini-beta-B16     | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    |                                        |
| quality_compute/grok-3-mini-beta-B32     | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    |                                        |
| quality_compute/grok-3-mini-beta-B8      | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    |                                        |
| grok-3-mini-beta                         | grok            | 5            | 20 / 20      | 90.00%                          | 100%                          | 2                               | 0                      | 0                    |                                        |
| o3                                       | openai          | 5            | 20 / 20      | 85.00%                          | 100%                          | 1                               | 2                      | 0                    | Corrected grading applied.             |
| quality_compute/grok-3-mini-fast-beta-B64| quality_compute | 5            | 20 / 20      | 85.00%                          | 100%                          | 1                               | 0                      | 2                    |                                        |
| gemini-2.5-pro-exp-03-25                 | gemini          | 10           | 20 / 20      | 65.00%                          | 100%                          | 0                               | 7                      | 0                    | March release                          |
| gemini-2.5-flash-preview-05-20           | gemini          | 8            | 20 / 20      | 80.00%                          | 100%                          | 4                               | 0                      | 0                    | NEW FLASH DATA! Significant jump.      |
| gemini-2.0-flash-thinking-exp-1219       | gemini          | 10           | 20 / 20      | 35.00%                          | 100%                          | 4                               | 8                      | 1                    |                                        |
| quality_compute/o4-mini-B8               | quality_compute | 5            | 20 / 20      | 30.00%                          | 100%                          | 12                              | 2                      | 0                    |                                        |
| gemini-2.5-pro-preview-05-06             | gemini          | 5            | 20 / 20      | 30.00%                          | 100%                          | 13                              | 1                      | 0                    | New data (May I/O); Regression         |
| deepseek-chat-B1                         | quality_compute | 5            | 20 / 20      | 25.00%                          | 100%                          | 10                              | 1                      | 4                    |                                        |
| o4-mini                                  | openai          | 5            | 20 / 20      | 20.00%                          | 100%                          | 13                              | 3                      | 0                    |                                        |
| deepseek-reasoner-B1                     | quality_compute | 5            | 20 / 20      | 10.00%                          | 100%                          | 17                              | 1                      | 0                    |                                        |
| gemini-2.0-flash-lite                    | gemini          | 10           | 20 / 20      | 10.00%                          | 100%                          | 0                               | 18                     | 0                    |                                        |
| gpt-4.1-nano                             | openai          | 5            | 20 / 20      | 5.00%                           | 100%                          | 0                               | 19                     | 0                    |                                        |
| quality_compute/gpt-4.1-nano-B32         | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 19                              | 0                      | 0                    |                                        |
| quality_compute/gpt-4.1-nano-B64         | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 19                              | 0                      | 0                    |                                        |
| quality_compute/grok-3-beta-B64          | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 19                              | 0                      | 0                    |                                        |
| quality_compute/gpt-4.1-mini-B32         | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 12                              | 3                      | 4                    |                                        |
| quality_compute/gpt-4.1-mini-B64         | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 12                              | 1                      | 6                    |                                        |
| grok-3-fast-latest                       | grok            | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |                                        |
| gemini-1.5-flash-8b                      | gemini          | 10           | 20 / 20      | 0.00%                           | 0%                            | 0                               | 20                     | 0                    |                                        |
| gemini-2.0-flash                         | gemini          | 10           | 20 / 20      | 0.00%                           | 0%                            | 0                               | 20                     | 0                    |                                        |
| gpt-4.1                                  | openai          | 30           | 20 / 20      | 0.00%                           | 0%                            | 0                               | 20                     | 0                    |                                        |
| quality_compute/gpt-4.1-nano-B128        | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |                                        |
| quality_compute/gpt-4.1-nano-B8          | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |                                        |
| quality_compute/grok-3-fast-beta-B16     | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |                                        |
| quality_compute/grok-3-fast-beta-B32     | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |                                        |
| quality_compute/grok-3-fast-beta-B8      | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |                                        |
| quality_compute/gpt-4.1-mini-B128        | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 9                               | 4                      | 7                    |                                        |
| quality_compute/gpt-4.1-mini-B8          | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 16                              | 1                      | 3                    |                                        |
| gpt-4.1-mini                             | openai          | 5            | 20 / 20      | 0.00%                           | 0%                            | 3                               | 13                     | 4                    |                                        |

**Commentary:** Tic-Tac-Toe remains a strong differentiator. `grok-3-mini-beta` and its QC variants dominate. The new `gemini-2.5-flash-preview-05-20` shows a massive jump to 80% `pass@1`. The new `gemini-2.5-pro-preview-05-06` (May I/O) shows a notable regression to 30% `pass@1` from the March version's 65%, indicating challenges for some "thinking" models in this strategic, rule-bound task.

### Task: Complex File Organizer
*   **Rounds:** Typically 10 or 35 turns max.
*   *Tests complex instruction following, state tracking, conditional logic, command generation.*

| Model                                  | Provider        | Turn Horizon | Attempts Run | `pass@1` (Perfect Match) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes                                  |
| :------------------------------------- | :-------------- | :----------- | :----------- | :----------------------- | :---------------------------- | :------------------------------ | :--------------------- | :------------------- | :------------------------------------- |
| grok-3-fast-beta                       | grok            | 35           | 20 / 20      | 100.00%                  | 100%                          | 0                               | 0                      | 0                    |                                        |
| gpt-4.1                                | openai          | 30           | 20 / 20      | 100.00%                  | 100%                          | 0                               | 0                      | 0                    |                                        |
| gpt-4.1-mini                           | openai          | 30           | 20 / 20      | 85.00%                   | 100%                          | 2                               | 1                      | 0                    |                                        |
| gemini-2.0-flash                       | gemini          | 35           | 20 / 20      | 85.00%                   | 100%                          | 2                               | 0                      | 1                    |                                        |
| gemini-2.5-flash-preview-05-20         | gemini          | 35           | 20 / 20      | 70.00%                   | 100%                          | 6                               | 0                      | 0                    | NEW FLASH DATA! Improvement.           |
| o4-mini                                | openai          | 30           | 20 / 20      | 70.00%                   | 100%                          | 6                               | 0                      | 0                    |                                        |
| gemini-2.0-flash-lite                  | gemini          | 35           | 20 / 20      | 65.00%                   | 100%                          | 1                               | 0                      | 6                    |                                        |
| gemini-2.5-flash-preview-04-17         | gemini          | 35           | 20 / 20      | 65.00%                   | 100%                          | 7                               | 0                      | 0                    |                                        |
| o3                                     | openai          | 30           | 20 / 20      | 55.00%                   | 100%                          | 8                               | 1                      | 0                    |                                        |
| quality_compute/grok-3-mini-beta-B8    | quality_compute | 35           | 20 / 20      | 45.00%                   | 100%                          | 11                              | 0                      | 0                    |                                        |
| deepseek-chat                          | deepseek        | 35           | 20 / 20      | 20.00%                   | 100%                          | 9                               | 7                      | 0                    | (QC provider in original table)        |
| gemini-2.5-pro-preview-05-06           | gemini          | 30           | 20 / 20      | 20.00%                   | 100%                          | 16                              | 0                      | 0                    | New data (May I/O); Improvement        |
| gemini-2.5-pro-exp-03-25               | gemini          | 35           | 20 / 20      | 15.00%                   | 100%                          | 17                              | 0                      | 0                    | March release                          |
| grok-3-mini-beta                       | grok            | 35           | 20 / 20      | 15.00%                   | 100%                          | 12                              | 5                      | 0                    |                                        |
| gemini-1.5-flash-8b                    | gemini          | 35           | 20 / 20      | 0.00%                    | 0%                            | 8                               | 0                      | 12                   |                                        |
| gpt-4.1-nano                           | openai          | 30           | 20 / 20      | 0.00%                    | 0%                            | 3                               | 17                     | 0                    |                                        |
| quality_compute/deepseek-reasoner      | quality_compute | 35           | 20 / 20      | 0.00%                    | 0%                            | 2                               | 18                     | 0                    |                                        |

**Commentary:** File System tests precise execution over a long time horizon. `grok-3-fast-beta` and `gpt-4.1` lead with perfect scores. The new `gemini-2.5-flash-preview-05-20` improved to 70% `pass@1` from its previous iteration's 65%. Non thinking models excel at this task while thinking models seem to struggle. This can be abated through more RL compute or test time compute.

### Task: Solar System Generator (Iterative Refinement)
*   **Rounds:** Typically 3 turns max.
*   *Tests HTML/JS code generation, incorporating visual feedback, debugging from logs. Now includes a Differentiated Score for better nuance, weighting partial successes at 0.3.*

| Model                               | Provider        | Turn Horizon | Attempts Run | `pass@1` (>5 Planets + Moon) | Differentiated Score (S3=1pt, S2=0.3pt) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes                                        |
| :---------------------------------- | :-------------- | :----------- | :----------- | :--------------------------- | :------------------------------------- | :------------------------------ | :--------------------- | :------------------- | :------------------------------------------- |
| gemini-2.5-pro-preview-05-06        | gemini          | 3            | 20 / 20      | 25.00%                       | 43.00%                                 | 12                              | 2                      | 0                    | New SOTA on Differentiated Score!            |
| o3                                  | openai          | 3            | 20 / 20      | 20.00%                       | 30.50%                                 | 7                               | 9                      | 0                    | Previous SOTA on pass@1.                     |
| gemini-2.5-pro-exp-03-25            | gemini          | 3            | 20 / 20      | 15.00%                       | 34.50%                                 | 13                              | 0                      | 0                    | March release.                               |
| deepseek-reasoner-B1                | quality_compute | 3            | 20 / 20      | 5.00%                        | 21.50%                                 | 11                              | 8                      | 0                    |                                              |
| quality_compute/grok-3-mini-beta-B8 | quality_compute | 3            | 20 / 20      | 5.00%                        | 21.50%                                 | 11                              | 8                      | 0                    |                                              |
| grok-3-beta                         | grok            | 3            | 20 / 20      | 0.00%                        | 22.50%                                 | 15                              | 5                      | 0                    |                                              |
| grok-3-mini-beta                    | grok            | 3            | 20 / 20      | 0.00%                        | 16.50%                                 | 11                              | 9                      | 0                    |                                              |
| gemini-2.5-flash-preview-05-20      | gemini          | 3            | 20 / 20      | 0.00%                        | 13.50%                                 | 9                               | 11                     | 0                    | NEW FLASH DATA!                              |
| deepseek-chat-B1                    | quality_compute | 3            | 20 / 20      | 0.00%                        | 9.00%                                  | 6                               | 14                     | 0                    |                                              |
| gpt-4.1                             | openai          | 3            | 20 / 20      | 0.00%                        | 1.50%                                  | 1                               | 19                     | 0                    |                                              |
| o4-mini                             | openai          | 3            | 20 / 20      | 0.00%                        | 7.50%                                  | 5                               | 15                     | 0                    |                                              |
| gpt-4.1-mini                        | openai          | 3            | 20 / 20      | 0.00%                        | 0.00%                                  | 0                               | 20                     | 0                    |                                              |
| gpt-4.1-nano                        | openai          | 3            | 20 / 20      | 0.00%                        | 0.00%                                  | 0                               | 20                     | 0                    |                                              |

**Commentary:** Iterative refinement remains a major hurdle. The **Differentiated Score** highlights models with partial success, with `gemini-2.5-pro-preview-05-06` leading at 43.00%. The new `gemini-2.5-flash-preview-05-20` shows improvement with 13.50%, indicating better iterative capabilities compared to its predecessor.

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
python toybench_cli.py --task solar_gen --provider gemini --model gemini-2.5-pro-preview-05-06 --attempts 20 --rounds 3

# --- Run with a Best of N variant (example using quality_compute) ---
python toybench_cli.py --task tic_tac_toe --provider quality_compute --model grok-3-mini-beta-B8 --attempts 20 --rounds 5

# --- Debug Run ---
python toybench_cli.py --task solar_gen --attempts 1 --rounds 3 --log_level DEBUG