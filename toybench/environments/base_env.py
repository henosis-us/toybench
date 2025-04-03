from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """Abstract Base Class for ToyBench Environments."""

    @abstractmethod
    def reset(self) -> str:
        """
        Resets the environment to its initial state.
        Returns:
            str: A string representation of the initial environment state, suitable for the LLM.
        """
        pass

    @abstractmethod
    def get_state(self) -> str:
        """
        Returns a string representation of the current environment state.
        Deprecated in favor of get_prompt_context, but kept for potential direct use
        or internal logic. Consider phasing out if only get_prompt_context is used externally.

        Returns:
            str: String representation of the current state.
        """
        pass

    @abstractmethod
    def get_goal(self) -> str:
        """
        Returns a string describing the high-level goal of the task in this environment.

        Returns:
            str: The goal description.
        """
        pass

    @abstractmethod
    def validate_action(self, action: str) -> bool:
        """
        Checks if the proposed action string is valid according to the environment's
        rules in the current state.

        Args:
            action (str): The action proposed by the agent.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        pass

    @abstractmethod
    def step(self, action: str) -> tuple[str, bool]:
        """
        Executes the validated action, updates the environment's internal state,
        and determines if the environment has reached a terminal state.

        Args:
            action (str): The validated action string to execute.

        Returns:
            tuple[str, bool]: A tuple containing:
                - str: The new state description after the action.
                - bool: True if the environment is in a terminal state (goal reached,
                        failed state, or max steps potentially implied), False otherwise.

        Raises:
            ValueError: If the action is invalid (though pre-validation is expected).
            Exception: For other internal environment errors during execution.
        """
        pass

    @abstractmethod
    def check_goal_achieved(self) -> bool:
        """
        Checks if the current internal state of the environment satisfies the
        specific goal criteria defined for the task. Called internally or after the loop.

        Returns:
            bool: True if the goal is achieved, False otherwise.
        """
        pass

    @abstractmethod
    def assess_intermediate_status(self) -> any:
        """
        Returns a representation of the current progress or state quality relative
        to the goal. Used for regression tracking. The returned value must support
        comparison (e.g., via <, > operators) where a 'better' state compares
        as greater than a 'worse' state.

        Returns:
            any: A comparable status object (e.g., int score, tuple) or None if
                 intermediate assessment is not applicable/implemented.
        """
        pass

    @abstractmethod
    def get_final_eval_input(self) -> str:
        """
        Returns the specific input required by the final evaluator LLM. This might be
        the final state description, a summary, or potentially other data like a
        file path for visual tasks.

        Returns:
            str: The input string for the final evaluation prompt.
        """
        pass

    @abstractmethod
    def get_prompt_context(self) -> dict:
        """
        Returns a dictionary containing all necessary context variables required
        to format the agent's action generation prompt template for this environment
        in its current state.

        This dictionary MUST include keys like 'goal' and 'current_state'.
        Specific environments can add other relevant keys (e.g., 'current_player',
        'available_actions_list').

        Returns:
            dict: A dictionary of context variables for prompt formatting.
        """
        pass

    @abstractmethod
    def get_agent_player_mark(self) -> str | None:
        """
        Returns the specific mark or identifier used by the agent being benchmarked
        within this environment, if applicable (e.g., 'X' in TicTacToe).
        Returns None if the concept isn't relevant for the environment.
        """
        pass