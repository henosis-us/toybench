# Updated README.md with new data from reports

# ToyBench: Evaluating Agentic Planning, Action, and Iterative Refinement
## Vision & Goals
**ToyBench** evaluates the capability of Large Language Models (LLMs) to **plan, take actions agentically over time, and refine outputs based on feedback**. Our vision is to create a method for measuring how well LLMs translate complex requirements and high-level goals into tangible, **"valid" sequences of actions or artifacts** within defined environments, assessed through a practical, multi-step process that may involve interaction and iteration.
Benchmark runs across **Tic-Tac-Toe** (strategic planning, multi-turn horizon), **File System** management (complex instruction following, state tracking, high-round horizon), and iterative **Solar System** generation (visual refinement, code generation, multimodal feedback, limited-round horizon) demonstrate significant variation in capabilities between different models and providers and highlight the distinct challenges posed by each task type and time horizon.
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
*   **Overall ToyBench Score:** Can be calculated as the **Average `pass@1` score across all included tasks** *if run with a consistent round limit*. The results presented below use different effective horizons per task (TTT: Unlimited, FS: 35, SolarGen: 3), making a single overall score less directly comparable. Performance should be assessed considering the specific task constraints and relative strengths.
## Overall Performance Summary (Varied Horizons)
This table summarizes the performance of models tested across Tic-Tac-Toe, FileSystem, and SolarGen. The "Overall Score" represents the average `pass@1` across these three tasks based on available 20/20 runs. Models are only included if they have a completed run (20 attempts) for all three tasks. Grok "fast" variants are grouped with their base models for this summary. The table has been updated with new data from recent runs, adding models like `o3` and `o4-mini`.

| Model                            | Provider        | Overall Score (Avg. `pass@1`) | Notes                                                                                                                               |
| :------------------------------- | :-------------- | :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| o3                               | openai          | **53.33%**                    | Strong performance across tasks.                                                                             |
| grok-3-mini-beta                 | grok            | **35.00%**                    | Excellent on Tic-Tac-Toe (90%), but struggled on FileSystem (15%) and SolarGen (0%).                                                |
| gemini-2.5-flash-preview-04-17   | gemini          | **35.00%**                    | Strong on FileSystem (65%), moderate on Tic-Tac-Toe (40%), failed on SolarGen.                                                      |
| gemini-2.5-pro-exp-03-25         | gemini          | **31.67%**                    | Good strategic planning (TTT 65%), capable iterative refinement (SolarGen 15%), but weak on FS (15%).                              |
| o4-mini                          | openai          | **30.00%**                    | Good on FileSystem (70%), but weak on Tic-Tac-Toe and SolarGen.                                                                     |
| gpt-4.1-mini                     | openai          | **28.33%**                    | Strong on FileSystem (85%), but failed on Tic-Tac-Toe and SolarGen.                                                                 |
| deepseek-chat-B1                 | quality_compute | **15.00%**                    | Moderate on Tic-Tac-Toe (25%) and FileSystem (20%), but failed on SolarGen.                                                         |
| deepseek-reasoner-B1             | quality_compute | **5.00%**                     | Some success on SolarGen (5%), but weak on Tic-Tac-Toe (10%) and FileSystem (0%).                                                   |
| gpt-4.1-nano                     | openai          | **1.67%**                     | Generally low `pass@1` across tasks, with minimal success observed.                                                                 |
| gemini-1.5-flash-8b              | gemini          | **0.00%**                     | Failed to achieve strict success (Score 3) on any task in these tests.                                                              |

**Notes:**
* The benchmark continues to differentiate model capabilities across diverse agentic challenges and horizons. New data highlights improvements in some models (e.g., `o3` showing strong performance) but reinforces that iterative tasks like SolarGen remain challenging.
* No single model excels universally. Grok and Gemini models show strong average performance, with Grok excelling in structured tasks and Gemini in iterative refinement for some variants.
* `grok-3-fast-beta` demonstrated exceptional proficiency in the FileSystem task (100% `pass@1`), while `grok-3-mini-beta` remains strong in Tic-Tac-Toe.
* Iterative tasks like SolarGen are still a weak point, with very few models achieving reliable success. `gemini-2.5-pro-exp` and `o3` are among the few with `pass@1` success.
* Reasoning-focused models poorly on long horizon tasks, across model families.
## Task-Specific Performance Details
Below are detailed results for each task, comparing all models tested under the specific round constraints reported in the logs, filtering for runs with 20/20 attempts unless noted otherwise. Tables have been updated with new data from the provided reports.
### Task: Tic-Tac-Toe (vs Optimal Opponent)
*   **Rounds:** Typically 5 or 10 turns max (implicitly unlimited until game ends in some runs).
*   *Tests strategic planning, rule adherence, and goal achievement.*
| Model                                    | Provider        | Turn Horizon | Attempts Run | `pass@1` (Win/Draw, No Invalid) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes |
| :--------------------------------------- | :-------------- | :----------- | :----------- | :------------------------------ | :---------------------------- | :------------------------------ | :--------------------- | :------------------- | :---- |
| quality_compute/grok-3-mini-beta-B128    | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    | New data added. |
| quality_compute/grok-3-mini-beta-B16     | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    | New data added. |
| quality_compute/grok-3-mini-beta-B32     | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    | New data added. |
| quality_compute/grok-3-mini-beta-B8      | quality_compute | 5            | 20 / 20      | 100.00%                         | 100%                          | 0                               | 0                      | 0                    | New data added. |
| grok-3-mini-beta                         | grok            | 5            | 20 / 20      | 90.00%                          | 100%                          | 2                               | 0                      | 0                    |       |
| o3                                       | openai          | 5            | 20 / 20      | 85.00%                          | 100%                          | 1                               | 2                      | 0                    | Corrected grading applied. |
| quality_compute/grok-3-mini-fast-beta-B64| quality_compute | 5            | 20 / 20      | 85.00%                          | 100%                          | 1                               | 0                      | 2                    | New data added. |
| gemini-2.5-pro-exp-03-25                 | gemini          | 10           | 20 / 20      | 65.00%                          | 100%                          | 0                               | 7                      | 0                    |       |
| gemini-2.5-flash-preview-04-17           | gemini          | 5            | 20 / 20      | 40.00%                          | 100%                          | 5                               | 0                      | 7                    |       |
| gemini-2.0-flash-thinking-exp-1219       | gemini          | 10           | 20 / 20      | 35.00%                          | 100%                          | 4                               | 8                      | 1                    | New data added. |
| deepseek-chat-B1                         | quality_compute | 5            | 20 / 20      | 25.00%                          | 100%                          | 10                              | 1                      | 4                    |       |
| o4-mini                                  | openai          | 5            | 20 / 20      | 20.00%                          | 100%                          | 13                              | 3                      | 0                    |       |
| deepseek-reasoner-B1                     | quality_compute | 5            | 20 / 20      | 10.00%                          | 100%                          | 17                              | 1                      | 0                    | New data added. |
| gemini-2.0-flash-lite                    | gemini          | 10           | 20 / 20      | 10.00%                          | 100%                          | 0                               | 18                     | 0                    |       |
| gpt-4.1-nano                             | openai          | 5            | 20 / 20      | 5.00%                           | 100%                          | 0                               | 19                     | 0                    |       |
| quality_compute/gpt-4.1-nano-B32         | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 19                              | 0                      | 0                    |       |
| quality_compute/gpt-4.1-nano-B64         | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 19                              | 0                      | 0                    |       |
| quality_compute/grok-3-beta-B64          | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 19                              | 0                      | 0                    |       |
| quality_compute/gpt-4.1-mini-B32         | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 12                              | 3                      | 4                    |       |
| quality_compute/gpt-4.1-mini-B64         | quality_compute | 5            | 20 / 20      | 5.00%                           | 100%                          | 12                              | 1                      | 6                    |       |
| quality_compute/o4-mini-B8               | quality_compute | 5            | 20 / 20      | 30.00%                          | 100%                          | 12                              | 2                      | 0                    |       |
| grok-3-fast-latest                       | grok            | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |       |
| gemini-1.5-flash-8b                      | gemini          | 10           | 20 / 20      | 0.00%                           | 0%                            | 0                               | 20                     | 0                    |       |
| gemini-2.0-flash                         | gemini          | 10           | 20 / 20      | 0.00%                           | 0%                            | 0                               | 20                     | 0                    |       |
| gpt-4.1                                  | openai          | 30           | 20 / 20      | 0.00%                           | 0%                            | 0                               | 20                     | 0                    |       |
| quality_compute/gpt-4.1-nano-B128        | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |       |
| quality_compute/gpt-4.1-nano-B8          | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |       |
| quality_compute/grok-3-fast-beta-B16     | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |       |
| quality_compute/grok-3-fast-beta-B32     | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |       |
| quality_compute/grok-3-fast-beta-B8      | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 20                              | 0                      | 0                    |       |
| quality_compute/gpt-4.1-mini-B128        | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 9                               | 4                      | 7                    |       |
| quality_compute/gpt-4.1-mini-B8          | quality_compute | 5            | 20 / 20      | 0.00%                           | 0%                            | 16                              | 1                      | 3                    |       |
| gpt-4.1-mini                             | openai          | 5            | 20 / 20      | 0.00%                           | 0%                            | 3                               | 13                     | 4                    |       |

**Commentary:** Tic-Tac-Toe requires robust strategic planning and perfect adherence to game rules. `grok-3-mini-beta` and its quality_compute variants continue to demonstrate outstanding performance. New data for models like `gemini-2.0-flash-thinking-exp-1219` shows moderate improvements, but many models still struggle with consistent wins against an optimal opponent.
### Task: Complex File Organizer
*   **Rounds:** Typically 10 or 35 turns max.
*   *Tests complex instruction following, state tracking, conditional logic, command generation.*
| Model                                  | Provider        | Turn Horizon | Attempts Run | `pass@1` (Perfect Match) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes |
| :------------------------------------- | :-------------- | :----------- | :----------- | :----------------------- | :---------------------------- | :------------------------------ | :--------------------- | :------------------- | :---- |
| grok-3-fast-beta                       | grok            | 35           | 20 / 20      | 100.00%                  | 100%                          | 0                               | 0                      | 0                    |       |
| gpt-4.1                                | openai          | 30           | 20 / 20      | 100.00%                  | 100%                          | 0                               | 0                      | 0                    |       |
| gpt-4.1-mini                           | openai          | 30           | 20 / 20      | 85.00%                   | 100%                          | 2                               | 1                      | 0                    |       |
| gemini-2.0-flash                       | gemini          | 35           | 20 / 20      | 85.00%                   | 100%                          | 2                               | 0                      | 1                    |       |
| o4-mini                                | openai          | 30           | 20 / 20      | 70.00%                   | 100%                          | 6                               | 0                      | 0                    |       |
| gemini-2.0-flash-lite                  | gemini          | 35           | 20 / 20      | 65.00%                   | 100%                          | 1                               | 0                      | 6                    |       |
| gemini-2.5-flash-preview-04-17         | gemini          | 35           | 20 / 20      | 65.00%                   | 100%                          | 7                               | 0                      | 0                    |       |
| o3                                     | openai          | 30           | 20 / 20      | 55.00%                   | 100%                          | 8                               | 1                      | 0                    |       |
| quality_compute/grok-3-mini-beta-B8    | quality_compute | 35           | 20 / 20      | 45.00%                   | 100%                          | 11                              | 0                      | 0                    | New data added. |
| deepseek-chat                      | quality_compute | 35           | 20 / 20      | 20.00%                   | 100%                          | 9                               | 7                      | 0                    |       |
| gemini-2.5-pro-exp-03-25               | gemini          | 35           | 20 / 20      | 15.00%                   | 100%                          | 17                              | 0                      | 0                    |       |
| grok-3-mini-beta                       | grok            | 35           | 20 / 20      | 15.00%                   | 100%                          | 12                              | 5                      | 0                    |       |
| gemini-1.5-flash-8b                    | gemini          | 35           | 20 / 20      | 0.00%                    | 0%                            | 8                               | 0                      | 12                   |       |
| gpt-4.1-nano                           | openai          | 30           | 20 / 20      | 0.00%                    | 0%                            | 3                               | 17                     | 0                    |       |
| quality_compute/deepseek-reasoner   | quality_compute | 35           | 20 / 20      | 0.00%                    | 0%                            | 2                               | 18                     | 0                    |       |

**Commentary:** File System tests precise execution of complex, stateful instructions. Top models like `grok-3-fast-beta` and `gpt-4.1` achieve perfect scores, while new data for variants like `grok-3-mini-beta-B8` shows moderate performance. Many models achieve partial success, indicating understanding but inconsistent execution.
### Task: Solar System Generator (Iterative Refinement)
*   **Rounds:** Typically 3 turns max.
*   *Tests HTML/JS code generation, incorporating  visual feedback, debugging from logs.*
| Model                            | Provider        | Turn Horizon | Attempts Run | `pass@1` (>5 Planets + Moon) | `pass@20` (Any Success in 20) | Partial Success (Score 2) Count | Failed (Score 1) Count | Failed (Error) Count | Notes |
| :------------------------------- | :-------------- | :----------- | :----------- | :--------------------------- | :---------------------------- | :------------------------------ | :--------------------- | :------------------- | :---- |
| o3                               | openai          | 3            | 20 / 20      | 20.00%                       | 100%                          | 7                               | 9                      | 0                    | New data added. |
| gemini-2.5-pro-exp-03-25         | gemini          | 3            | 20 / 20      | 15.00%                       | 100%                          | 13                              | 0                      | 0                    |       |
| deepseek-reasoner-B1             | quality_compute | 3            | 20 / 20      | 5.00%                        | 100%                          | 11                              | 8                      | 0                    | New data added. |
| grok-3-beta                      | grok            | 3            | 20 / 20      | 0.00%                        | 0%                            | 15                              | 5                      | 0                    |       |
| grok-3-mini-beta                 | grok            | 3            | 20 / 20      | 0.00%                        | 0%                            | 11                              | 9                      | 0                    |       |
| deepseek-chat-B1                 | quality_compute | 3            | 20 / 20      | 0.00%                        | 0%                            | 6                               | 14                     | 0                    |       |
| gemini-2.5-flash-preview-04-17   | gemini          | 3            | 20 / 20      | 0.00%                        | 0%                            | 1                               | 19                     | 0                    |       |
| gpt-4.1-mini                     | openai          | 3            | 20 / 20      | 0.00%                        | 0%                            | 0                               | 20                     | 0                    |       |
| gpt-4.1-nano                     | openai          | 3            | 20 / 20      | 0.00%                        | 0%                            | 0                               | 20                     | 0                    |       |
| o4-mini                          | openai          | 3            | 20 / 20      | 0.00%                        | 0%                            | 5                               | 15                     | 0                    |       |
| gpt-4.1                          | openai          | 3            | 20 / 20      | 0.00%                        | 0%                            | 1                               | 19                     | 0                    |       |

**Commentary:** Iterative tasks with limited turns and complex feedback remain challenging. `gemini-2.5-pro-exp` and `o3` are among the few models with reliable `pass@1` success. New data for `deepseek-reasoner-B1` shows minor improvements, but most models achieve partial success without meeting full criteria.
## Prerequisites & Usage
**Prerequisites:**
*   **Python:** Version 3.9+ recommended.
*   **Dependencies:** Install required packages:    
    ```bash    pip install google-generativeai python-dotenv requests selenium webdriver-manager    ```
*   **API Keys:** Set `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `QUALITY_COMPUTE_API_KEY` in environment or `.env` file as needed for the providers you wish to test.
*   **WebDriver (for `solar_gen`):** Install Chrome/Chromium and ensure `chromedriver` (matching version) is in PATH or managed by `webdriver-manager`.
**Usage Examples:**
```bash# --- Run TicTacToe ---# Use --rounds to set the max turn horizon if desired (e.g., --rounds 5)python toybench_cli.py --task tic_tac_toe --provider grok --model grok-3-mini-beta --attempts 20 --rounds 5

# --- Run Complex File System ---# Use --rounds 50 or similar for a high cap if desired (e.g., --rounds 35 or --rounds 50)python toybench_cli.py --task file_system --provider openai --model gpt-4.1 --attempts 20 --rounds 30

# --- Run Solar System Generator ---# Use --rounds 3 or similar for a limited horizonpython toybench_cli.py --task solar_gen --provider gemini --model gemini-2.5-pro-exp-03-25 --attempts 20 --rounds 3

# --- Debug Run ---python toybench_cli.py --task solar_gen --attempts 1 --rounds 3 --log_level DEBUG