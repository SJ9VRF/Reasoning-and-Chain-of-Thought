# Implements Self-Consistency for Chain of Thought.

# src/self_consistency.py

from collections import Counter
import matplotlib.pyplot as plt
from llm_interface import LLMInterface
from tools import WikipediaTool

class SelfConsistency:
    """
    Implements Self-Consistency for improving the reliability of LLM responses.
    """

    def __init__(self, llm: LLMInterface, tool: WikipediaTool):
        """
        Initialize the SelfConsistency instance.

        Args:
            llm (LLMInterface): An instance of LLMInterface.
            tool (WikipediaTool): An instance of WikipediaTool.
        """
        self.llm = llm
        self.tool = tool

    def run_multiple_responses(self, prompt: str, parameters: dict, runs: int = 40) -> Counter:
        """
        Generate multiple responses and count their occurrences.

        Args:
            prompt (str): The prompt to send to the LLM.
            parameters (dict): Parameters for the LLM call.
            runs (int): Number of times to run the prompt.

        Returns:
            Counter: A counter of the different answers.
        """
        responses = []
        answers = []

        for i in range(runs):
            print(f"Response {i + 1}...")
            response = self.llm.call_llm(prompt, show_activity=False)
            responses.append(response)
            try:
                answer = response.split("The answer is")[1].split(".")[0].strip()
            except Exception:
                answer = "NA"
            answers.append(answer)
            print(response)

        answer_counts = Counter(answers)
        print("Answers and counts from most common to least common:")
        print(answer_counts.most_common())
        return answer_counts

    def plot_answer_distribution(self, answer_counts: Counter):
        """
        Plot the distribution of answers.

        Args:
            answer_counts (Counter): A counter of the different answers.
        """
        fig, ax = plt.subplots()
        ax.bar(answer_counts.keys(), answer_counts.values())
        ax.tick_params(axis='x', rotation=55)
        plt.show()
