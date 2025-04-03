import logging
import re
import random # Still useful if minimax returns multiple best moves (though unlikely here)
import math # For infinity in minimax
from .base_env import BaseEnvironment

logger = logging.getLogger(__name__)

class TicTacToeEnv(BaseEnvironment):
    """
    Tic-Tac-Toe game environment where the agent plays as 'X' and the
    environment script plays optimally for 'O' using Minimax.
    Evaluation is deterministic. Regression tracking is disabled.
    """
    def __init__(self, goal_description: str):
        self._goal = goal_description
        self.agent_player_mark = 'X'
        self.opponent_player_mark = 'O'
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = self.agent_player_mark
        self.winner = None
        self.is_draw = False

    def reset(self) -> str:
        """Resets the environment to the initial state."""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = self.agent_player_mark
        self.winner = None
        self.is_draw = False
        logger.info("Tic Tac Toe Environment Reset. Agent ('X') starts vs Optimal Opponent ('O').")
        return self._get_state_string()

    def _render_board(self) -> str:
        """Formats the board into a string."""
        rows = [" | ".join(row) for row in self.board]
        separator = "\n" + "---+---+---" + "\n"
        return separator.join(rows)

    def _get_state_string(self) -> str:
        """Generates the state description string."""
        board_str = self._render_board()
        if self.winner: status = f"Game Over. Player {self.winner} wins!"
        elif self.is_draw: status = "Game Over. It's a draw!"
        else: status = f"Player {self.current_player}'s turn."
        return f"Tic-Tac-Toe Board:\n{board_str}\n\nStatus: {status}"

    # --- Helper functions for board state analysis (used by Minimax) ---
    def _get_valid_moves_on_board(self, board_state) -> list[tuple[int, int]]:
        """Returns list of empty (row, col) tuples for a given board state."""
        moves = []
        for r in range(3):
            for c in range(3):
                if board_state[r][c] == ' ':
                    moves.append((r, c))
        return moves

    def _check_win_on_board(self, board_state, player_mark) -> bool:
        """Checks if the specified player has won on the given board state."""
        b = board_state; p = player_mark
        for i in range(3):
            if all(b[i][j] == p for j in range(3)) or all(b[j][i] == p for j in range(3)): return True
        if all(b[i][i] == p for i in range(3)) or all(b[i][2-i] == p for i in range(3)): return True
        return False

    def _is_board_full(self, board_state) -> bool:
        """Checks if the given board state is full."""
        return all(board_state[r][c] != ' ' for r in range(3) for c in range(3))

    # --- Minimax Implementation ---
    def _minimax(self, board_state, is_maximizing_player) -> int:
        """
        Minimax algorithm to determine the score of a board state.
        Opponent 'O' is maximizing (+10 for win), Agent 'X' is minimizing (-10 for win).
        """
        # Check terminal states
        if self._check_win_on_board(board_state, self.opponent_player_mark): return 10
        if self._check_win_on_board(board_state, self.agent_player_mark): return -10
        if self._is_board_full(board_state): return 0 # Draw

        valid_moves = self._get_valid_moves_on_board(board_state)

        if is_maximizing_player: # Opponent 'O's turn in simulation
            best_score = -math.inf
            for r, c in valid_moves:
                board_state[r][c] = self.opponent_player_mark # Make move
                score = self._minimax(board_state, False) # Simulate minimizing player's turn next
                board_state[r][c] = ' ' # Undo move
                best_score = max(score, best_score)
            return best_score
        else: # Agent 'X's turn in simulation
            best_score = math.inf
            for r, c in valid_moves:
                board_state[r][c] = self.agent_player_mark # Make move
                score = self._minimax(board_state, True) # Simulate maximizing player's turn next
                board_state[r][c] = ' ' # Undo move
                best_score = min(score, best_score)
            return best_score

    def _find_best_move(self) -> tuple[int, int] | None:
        """
        Finds the optimal move for the current player ('O') using Minimax.
        """
        best_score = -math.inf
        best_move = None
        board_copy = [row[:] for row in self.board] # Work on a copy

        valid_moves = self._get_valid_moves_on_board(board_copy)
        if not valid_moves: return None # Should not happen if game not over

        for r, c in valid_moves:
            board_copy[r][c] = self.opponent_player_mark # Make the move for 'O'
            # Evaluate score assuming 'X' plays next (minimizing)
            score = self._minimax(board_copy, False)
            board_copy[r][c] = ' ' # Undo the move

            # 'O' wants to maximize the score
            if score > best_score:
                best_score = score
                best_move = (r, c)

        # Handle cases where multiple moves yield the same best score (optional: random choice among them)
        # For deterministic optimal, just return the first one found.
        if best_move is None and valid_moves: # Should only happen if all moves lead to same score? Fallback needed?
            logger.warning("Minimax couldn't definitively find a best move, picking first valid.")
            best_move = valid_moves[0]

        return best_move

    # --- Interface Methods Implementation --- (getters, validate remain same) ---

    def get_agent_player_mark(self) -> str | None: return self.agent_player_mark
    def get_state(self) -> str: return self._get_state_string()
    def get_goal(self) -> str: return self._goal
    def get_prompt_context(self) -> dict:
        return { "goal": self.get_goal(), "current_state": self._get_state_string(), "current_player": self.agent_player_mark }
    def validate_action(self, action: str) -> bool:
        if self.current_player != self.agent_player_mark: logger.warning(f"Validation called out of turn for {self.agent_player_mark}"); return False
        match = re.search(r"(\d+)\s*,\s*(\d+)", action) or re.search(r"row\s+(\d+)\s+col(?:umn)?\s+(\d+)", action, re.IGNORECASE)
        if not match: logger.debug(f"Could not parse coordinates: '{action}'"); return False
        try: r, c = int(match.group(1)) - 1, int(match.group(2)) - 1
        except ValueError: logger.debug(f"Invalid coord format: '{action}'"); return False
        if not (0 <= r < 3 and 0 <= c < 3): logger.debug(f"Coords out of bounds: {r+1},{c+1}"); return False
        if self.board[r][c] != ' ': logger.debug(f"Cell not empty: {r+1},{c+1}"); return False
        return True

    def step(self, action: str) -> tuple[str, bool]:
        """Executes a validated action. Assumes action matches self.current_player."""
        if self.winner or self.is_draw: logger.warning("Step called on finished game."); return self._get_state_string(), True

        match = re.search(r"(\d+)\s*,\s*(\d+)", action) or re.search(r"row\s+(\d+)\s+col(?:umn)?\s+(\d+)", action, re.IGNORECASE)
        if not match:
            logger.error(f"Invalid action format in step: '{action}'")
            self.winner = self.opponent_player_mark if self.current_player == self.agent_player_mark else self.agent_player_mark
            return self._get_state_string(), True
        r, c = int(match.group(1)) - 1, int(match.group(2)) - 1

        if self.board[r][c] != ' ':
             logger.error(f"Cell {r+1},{c+1} occupied in step: '{action}'.")
             self.winner = self.opponent_player_mark if self.current_player == self.agent_player_mark else self.agent_player_mark
             return self._get_state_string(), True

        player_making_move = self.current_player
        self.board[r][c] = player_making_move
        logger.debug(f"Step executed: Player {player_making_move} placed at {r+1},{c+1}")

        # Use internal methods now that check self.board
        self._check_current_win_condition()
        if not self.winner: self._check_current_draw_condition()

        terminal = self.winner is not None or self.is_draw

        if not terminal:
            self.current_player = self.opponent_player_mark if self.current_player == self.agent_player_mark else self.agent_player_mark
            logger.debug(f"Player switched to: {self.current_player}")
        else:
             logger.info(f"Game is terminal. Winner: {self.winner}, Draw: {self.is_draw}")

        return self._get_state_string(), terminal

    def make_opponent_move(self) -> tuple[str | None, str, bool]:
        """
        Chooses the OPTIMAL valid move for the opponent ('O') using Minimax,
        executes it via step, and returns action, new state, terminal status.
        """
        if self.current_player != self.opponent_player_mark:
            logger.error(f"make_opponent_move called, but it's not {self.opponent_player_mark}'s turn")
            return None, self._get_state_string(), True
        if self.winner or self.is_draw:
            logger.warning("make_opponent_move called on a finished game.")
            return None, self._get_state_string(), True

        # Find the best move using Minimax logic
        best_move_coords = self._find_best_move()

        if best_move_coords is None:
            # This case means no valid moves were found by _find_best_move, implies draw or error
            logger.error("Optimal opponent found no best move. Game should have ended?")
            if not self.winner and not self.is_draw: self.is_draw = True # Force draw state if somehow missed
            return None, self._get_state_string(), True

        row, col = best_move_coords
        action_string = f"place {self.opponent_player_mark} at {row+1},{col+1}"
        logger.info(f"Opponent ({self.opponent_player_mark}) chose optimal move: {action_string}")

        try:
             new_state_string, is_terminal = self.step(action_string)
             return action_string, new_state_string, is_terminal
        except Exception as e:
             logger.error(f"Error during opponent's optimal step execution: {e}", exc_info=True)
             return action_string, self._get_state_string(), True

    # Update internal win/draw checkers to use self.board and set self.winner/is_draw
    def _check_current_win_condition(self):
        """Checks if the player who just moved (self.current_player) has won."""
        if self._check_win_on_board(self.board, self.current_player):
             self.winner = self.current_player

    def _check_current_draw_condition(self):
        """Checks if the current board state is a draw."""
        if self.winner is None and self._is_board_full(self.board):
            self.is_draw = True

    # check_goal_achieved, assess_intermediate_status, get_final_eval_input remain the same
    def check_goal_achieved(self) -> bool:
        return self.winner == self.agent_player_mark
    def assess_intermediate_status(self) -> any:
        return None # Disable regression tracking
    def get_final_eval_input(self) -> str:
        return self._get_state_string()