import logging
import os
from typing import Tuple, Dict, Any, List, Set

from .base_env import BaseEnvironment

# Optional browser-based rendering support (headless Chrome)
try:
    from browser_utils import render_and_capture
    BROWSER_UTILS_AVAILABLE = True
except ImportError:
    render_and_capture = None
    BROWSER_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SokobanEnv(BaseEnvironment):
    """
    Sokoban environment with deterministic evaluation.

    - ASCII grid with walls '#', floor ' ', goals '.', boxes '$', box-on-goal '*',
      player '@', player-on-goal '+'.
    - Actions: one of {up, down, left, right, reset} per turn.
    - Deterministic scoring at the end: 3 if solved, 2 if partial progress, 1 otherwise.
    """

    # Bundled levels (trivial smoke test removed). Level 0 is the previous larger variant.
    # Design note: levels are enclosed by walls and use classic Sokoban symbols.
    LEVELS: List[List[str]] = [
        [
            "#######",
            "#  .  #",
            "#  $  #",
            "# .@  #",
            "#  $  #",
            "#     #",
            "#######",
        ],
        # HARD Level A: multiple boxes in corridors; requires careful ordering to avoid deadlocks
        [
            "#############",
            "#     #     #",
            "# .$#$$#$.  #",
            "#  #   #  ###",
            "### #.@.#   #",
            "#   #   # . #",
            "#  .$#$$#$. #",
            "#     #     #",
            "#############",
        ],
        # HARD Level B: compact maze with tight turns; goals scattered behind narrow gates
        [
            "###############",
            "#   .#   #.   #",
            "# $# # $ # #$ #",
            "#   ### ###   #",
            "### #  $  # ###",
            "# . # #@# # . #",
            "### #  $  # ###",
            "#   ### ###   #",
            "# $# # $ # #$ #",
            "#   .#   #.   #",
            "###############",
        ],
        # SIMPLE Two-push Level: small enclosed board requiring two upward pushes
        [
            "#####",
            "# . #",
            "#   #",
            "# $ #",
            "# @ #",
            "#####",
        ],
        # EASY Level: two boxes and two goals in a compact room
        [
            "#########",
            "#   .   #",
            "#  $ $  #",
            "#   @   #",
            "#   .   #",
            "#########",
        ],
        # MEDIUM Level: four boxes and four goals with central walls
        [
            "###########",
            "#  #   #  #",
            "# .$   $. #",
            "#   ###   #",
            "### #@# ###",
            "#   ###   #",
            "# .$   $. #",
            "#  #   #  #",
            "###########",
        ],
        # HARD Level: symmetrical puzzle with corridors and four goals
        [
            "#############",
            "# .   #   . #",
            "#  $ ### $  #",
            "#     @     #",
            "#  $ ### $  #",
            "# .   #   . #",
            "#############",
        ],
    ]

    # Brief notes aligned by index to LEVELS
    LEVEL_NOTES: List[str] = [
        "Base: compact room with 2 boxes and 2 goals",
        "Hard A: multiple boxes in corridors; careful ordering needed",
        "Hard B: compact maze with tight turns; scattered goals",
        "Simple: two-push vertical solution (tutorial-like)",
        "Easy: two boxes and two goals in a compact room",
        "Medium: four boxes with central walls; requires routing",
        "Hard: symmetrical corridors with four goals",
    ]

    @classmethod
    def get_level_metadata(cls) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        for i, level in enumerate(cls.LEVELS):
            h = len(level)
            w = len(level[0]) if h else 0
            # Count symbols from raw ASCII layout
            goals = sum(row.count('.') + row.count('*') + row.count('+') for row in level)
            boxes = sum(row.count('$') + row.count('*') for row in level)
            players = sum(row.count('@') + row.count('+') for row in level)
            note = cls.LEVEL_NOTES[i] if i < len(cls.LEVEL_NOTES) else ""
            meta.append({
                'index': i,
                'size': f"{h}x{w}",
                'goals': goals,
                'boxes': boxes,
                'players': players,
                'note': note,
            })
        return meta

    MOVE_DELTAS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    HTML_FILENAME = "sokoban_iteration_{turn}.html"
    SCREENSHOT_FILENAME = "sokoban_screenshot_iteration_{turn}.png"
    LOG_FILENAME = "sokoban_browser_logs_iteration_{turn}.txt"

    def __init__(self, goal_description: str, max_steps: int = 100, level_index: int = 0, output_dir_base: str | None = None):
        self._goal = goal_description
        self._max_steps = max_steps
        self._level_index = max(0, min(level_index, len(self.LEVELS) - 1))
        self._output_dir = output_dir_base

        # Dynamic state
        self.grid: List[List[str]] = []
        self.h: int = 0
        self.w: int = 0
        self.player: Tuple[int, int] | None = None
        self.goals: Set[Tuple[int, int]] = set()
        self.boxes: Set[Tuple[int, int]] = set()
        self.move_count: int = 0
        self.invalid_moves: int = 0
        self.solved: bool = False

        # Artifact state
        self.current_html_path: str | None = None
        self.current_screenshot_path: str | None = None
        self.current_browser_log_path: str | None = None

        self.reset()

    # --- BaseEnvironment required methods ---
    def reset(self) -> str:
        level = [list(row) for row in self.LEVELS[self._level_index]]
        self.grid = level
        self.h = len(level)
        self.w = len(level[0]) if self.h > 0 else 0
        self.goals.clear()
        self.boxes.clear()
        self.player = None
        self.move_count = 0
        self.invalid_moves = 0
        self.solved = False

        # Parse initial positions
        for r in range(self.h):
            for c in range(self.w):
                ch = self.grid[r][c]
                if ch == '.':
                    self.goals.add((r, c))
                elif ch == '*':
                    self.goals.add((r, c))
                    self.boxes.add((r, c))
                    self.grid[r][c] = ' '  # track boxes separately
                elif ch == '+':
                    self.goals.add((r, c))
                    self.player = (r, c)
                    self.grid[r][c] = ' '
                elif ch == '$':
                    self.boxes.add((r, c))
                    self.grid[r][c] = ' '
                elif ch == '@':
                    self.player = (r, c)
                    self.grid[r][c] = ' '

        if self.player is None:
            # Fallback: find any floor and set player
            for r in range(self.h):
                for c in range(self.w):
                    if self.grid[r][c] == ' ':
                        self.player = (r, c)
                        break
                if self.player:
                    break

        logger.info("SokobanEnv Reset. Level %d loaded. Board %dx%d.", self._level_index + 1, self.h, self.w)
        # Write initial artifact for inspection (turn 0) if output directory specified
        if self._output_dir:
            try:
                os.makedirs(self._output_dir, exist_ok=True)
                self._write_visual_artifacts(turn=0)
            except Exception as e:
                logger.warning(f"Failed to create initial Sokoban artifacts: {e}")

        return self._render_state()

    def get_state(self) -> str:
        return self._render_state()

    def get_goal(self) -> str:
        return self._goal

    def validate_action(self, action: str) -> bool:
        if not isinstance(action, str) or not action.strip():
            return False
        a = action.strip().lower()
        if a in ("up", "down", "left", "right", "reset"):
            return True
        # also allow forms like "move up"
        parts = a.split()
        if len(parts) == 2 and parts[0] == "move" and parts[1] in self.MOVE_DELTAS:
            return True
        return False

    def step(self, action: str) -> Tuple[str, bool]:
        if self.solved:
            return self._render_state(), True

        a = action.strip().lower()
        if a == "reset":
            logger.info("SokobanEnv: reset action received.")
            self.reset()
            return self._render_state(), False

        if a.startswith("move "):
            a = a.split()[1]

        if a not in self.MOVE_DELTAS:
            self.invalid_moves += 1
            logger.debug("Invalid action string: %s", action)
            return self._render_state(feedback="Invalid action. Use up/down/left/right/reset."), False

        dr, dc = self.MOVE_DELTAS[a]
        pr, pc = self.player
        nr, nc = pr + dr, pc + dc

        # Helper to check walls
        def is_wall(r: int, c: int) -> bool:
            return self.grid[r][c] == '#'

        # Inside bounds (level is walled, but guard anyway)
        if not (0 <= nr < self.h and 0 <= nc < self.w) or is_wall(nr, nc):
            self.invalid_moves += 1
            return self._render_state(feedback="Blocked by wall."), False

        # If moving into a box
        if (nr, nc) in self.boxes:
            br, bc = nr + dr, nc + dc
            if not (0 <= br < self.h and 0 <= bc < self.w) or is_wall(br, bc) or (br, bc) in self.boxes:
                self.invalid_moves += 1
                return self._render_state(feedback="Cannot push box."), False
            # Push the box
            self.boxes.remove((nr, nc))
            self.boxes.add((br, bc))
            self.player = (nr, nc)
            self.move_count += 1
        else:
            # Normal move
            self.player = (nr, nc)
            self.move_count += 1

        self.solved = self.check_goal_achieved()

        # Persist artifacts for this move number (use move_count as the turn index)
        if self._output_dir:
            try:
                self._write_visual_artifacts(turn=self.move_count)
            except Exception as e:
                logger.warning(f"Failed to write Sokoban artifacts for turn {self.move_count}: {e}")

        return self._render_state(), self.solved

    def check_goal_achieved(self) -> bool:
        # All goals must have a box
        return all(goal in self.boxes for goal in self.goals)

    def assess_intermediate_status(self) -> Any:
        boxes_on_goals = sum(1 for b in self.boxes if b in self.goals)
        # Favor more boxes on goals, then fewer moves
        return (boxes_on_goals, -self.move_count)

    def get_final_eval_input(self) -> str:
        """
        Returns a path to the latest screenshot (preferred for inspection).
        Falls back to ASCII board if no screenshot is available.
        """
        if isinstance(self.current_screenshot_path, str) and os.path.exists(self.current_screenshot_path):
            return self.current_screenshot_path
        return self._render_state(include_legend=False)

    def get_prompt_context(self) -> Dict[str, Any]:
        return {
            "goal": self.get_goal(),
            "current_state": self._render_state(),
            "available_moves": "up, down, left, right, reset",
        }

    def get_agent_player_mark(self) -> str | None:
        # Not used for Sokoban
        return None

    # --- Deterministic final evaluation ---
    def evaluate_final_state(self) -> int:
        if self.solved:
            return 3
        boxes_on_goals = sum(1 for b in self.boxes if b in self.goals)
        return 2 if boxes_on_goals > 0 else 1

    # --- Helpers ---
    def _render_state(self, feedback: str | None = None, include_legend: bool = True) -> str:
        # Compose a temporary board for rendering
        board = [row[:] for row in self.grid]
        # Place goals
        for (r, c) in self.goals:
            if board[r][c] == ' ':
                board[r][c] = '.'
        # Place boxes (on top of goals if applicable)
        for (r, c) in self.boxes:
            if (r, c) in self.goals:
                board[r][c] = '*'
            else:
                board[r][c] = '$'
        # Place player
        pr, pc = self.player
        if (pr, pc) in self.goals:
            board[pr][pc] = '+'
        else:
            board[pr][pc] = '@'

        header = []
        if include_legend:
            header.append("Legend: #: wall, .: goal, $: box, *: box-on-goal, @: player, +: player-on-goal")
            header.append(f"Moves: {self.move_count} | Invalid: {self.invalid_moves} | Boxes on goals: {sum(1 for b in self.boxes if b in self.goals)}/{len(self.goals)}")
            header.append(f"Solved: {self.solved}")
        if feedback:
            header.append(f"Feedback: {feedback}")

        grid_str = "\n".join("".join(row) for row in board)
        return ("\n".join(header) + ("\n" if header else "")) + grid_str

    def _write_visual_artifacts(self, turn: int):
        """Writes an HTML snapshot and attempts to capture a screenshot via headless browser."""
        html_path = os.path.join(self._output_dir, self.HTML_FILENAME.format(turn=turn))
        screenshot_path = os.path.join(self._output_dir, self.SCREENSHOT_FILENAME.format(turn=turn))
        log_path = os.path.join(self._output_dir, self.LOG_FILENAME.format(turn=turn))

        # Generate HTML content
        html = self._render_html_document()
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        # Attempt to capture screenshot (gracefully degrade if Selenium not available)
        if BROWSER_UTILS_AVAILABLE and callable(render_and_capture):
            ok, msg = render_and_capture(html_path, screenshot_path, log_path, browser_type="chrome")
            if not ok:
                logger.warning(f"Sokoban screenshot capture failed: {msg}")
        else:
            # Create empty placeholder files for consistency
            try:
                open(screenshot_path, 'a').close()
                open(log_path, 'a').close()
            except Exception as e:
                logger.debug(f"Failed to create placeholder artifacts: {e}")

        self.current_html_path = html_path
        self.current_screenshot_path = screenshot_path
        self.current_browser_log_path = log_path

    def _render_html_document(self) -> str:
        """Renders a simple self-contained HTML page visualizing the Sokoban board."""
        # Create a layer of chars similar to _render_state but as DOM
        board = [row[:] for row in self.grid]
        for (r, c) in self.goals:
            if board[r][c] == ' ':
                board[r][c] = '.'
        for (r, c) in self.boxes:
            board[r][c] = '*' if (r, c) in self.goals else '$'
        pr, pc = self.player
        board[pr][pc] = '+' if (pr, pc) in self.goals else '@'

        rows = []
        for r in range(self.h):
            cells = []
            for c in range(self.w):
                ch = board[r][c]
                cls = {
                    '#': 'wall',
                    ' ': 'floor',
                    '.': 'goal',
                    '$': 'box',
                    '*': 'box_goal',
                    '@': 'player',
                    '+': 'player_goal',
                }.get(ch, 'floor')
                label = ch if ch != ' ' else ''
                cells.append(f'<div class="cell {cls}">{label}</div>')
            rows.append('<div class="row">' + ''.join(cells) + '</div>')

        legend = (
            '<div class="legend">'
            '<span class="key"><span class="swatch wall"></span>Wall</span>'
            '<span class="key"><span class="swatch goal"></span>Goal</span>'
            '<span class="key"><span class="swatch box"></span>Box</span>'
            '<span class="key"><span class="swatch box_goal"></span>Box on Goal</span>'
            '<span class="key"><span class="swatch player"></span>Player</span>'
            '</div>'
        )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Sokoban Snapshot</title>
  <style>
    body {{ background:#111; color:#eee; font-family: Arial, sans-serif; padding: 16px; }}
    .board {{ display: inline-block; border: 2px solid #444; background:#222; padding: 8px; }}
    .row {{ display: flex; }}
    .cell {{ width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; }}
    .wall {{ background: #333; }}
    .floor {{ background: #1b1b1b; }}
    .goal {{ background: #4b3f00; box-shadow: inset 0 0 0 2px #ffd54d; }}
    .box {{ background: #8d5524; }}
    .box_goal {{ background: #6d8524; box-shadow: inset 0 0 0 2px #ffd54d; }}
    .player {{ background: #1e88e5; }}
    .player_goal {{ background: linear-gradient(135deg,#1e88e5 50%, #4b3f00 50%); box-shadow: inset 0 0 0 2px #ffd54d; }}
    .legend {{ margin-top: 12px; display: flex; gap: 16px; flex-wrap: wrap; }}
    .key {{ display: inline-flex; align-items: center; gap: 6px; color: #ccc; }}
    .swatch {{ width: 16px; height: 16px; display: inline-block; border: 1px solid #666; }}
    .swatch.wall {{ background:#333; }}
    .swatch.goal {{ background:#4b3f00; box-shadow: inset 0 0 0 2px #ffd54d; }}
    .swatch.box {{ background:#8d5524; }}
    .swatch.box_goal {{ background:#6d8524; box-shadow: inset 0 0 0 2px #ffd54d; }}
    .swatch.player {{ background:#1e88e5; }}
  </style>
  <script>
    // Log a small message for debug capture
    console.log('Sokoban snapshot rendered.');
  </script>
  </head>
<body>
  <h1 style="margin:0 0 12px 0; font-size:18px; color:#ddd;">Sokoban Snapshot</h1>
  <div class="board">{''.join(rows)}</div>
  {legend}
  <div style="margin-top:10px; color:#aaa; font-size:12px;">Moves: {self.move_count} | Boxes on goals: {sum(1 for b in self.boxes if b in self.goals)}/{len(self.goals)} | Solved: {str(self.solved).lower()}</div>
</body>
</html>
"""
        return html
