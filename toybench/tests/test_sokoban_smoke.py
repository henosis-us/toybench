import unittest

from environments.sokoban_env import SokobanEnv


class TestSokobanSimpleLevel(unittest.TestCase):
    def setUp(self):
        # Store original levels to restore after test
        self._orig_levels = SokobanEnv.LEVELS[:]

        # Simple enclosed board requiring 2 pushes (not solvable in 1 move)
        # Layout (5x5 interior inside walls):
        # #####
        # # . #   goal above box with one empty in between
        # #   #
        # # $ #   box here
        # # @ #   player below box
        # #####
        level = [
            "#####",
            "# . #",
            "#   #",
            "# $ #",
            "# @ #",
            "#####",
        ]

        SokobanEnv.LEVELS = [level]

    def tearDown(self):
        SokobanEnv.LEVELS = self._orig_levels

    def test_two_push_solution(self):
        env = SokobanEnv(goal_description="Push box to goal", max_steps=10, level_index=0, output_dir_base=None)
        state0 = env.get_state()
        self.assertIn("Solved: False", state0)
        self.assertEqual(env.move_count, 0)
        self.assertEqual(env.invalid_moves, 0)

        # Two valid pushes upward to reach the goal (requires >1 move)
        state1, done1 = env.step("up")
        self.assertFalse(done1)
        self.assertIn("Moves: 1", state1)
        self.assertEqual(env.move_count, 1)

        state2, done2 = env.step("up")
        self.assertTrue(done2)
        self.assertIn("Solved: True", state2)
        self.assertEqual(env.move_count, 2)

        # Deterministic evaluation should report full success
        self.assertEqual(env.evaluate_final_state(), 3)


if __name__ == "__main__":
    unittest.main()
