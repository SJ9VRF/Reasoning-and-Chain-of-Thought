# src/chain_of_thought.py

from llm_interface import LLMInterface
from tools import WikipediaTool

class ChainOfThought:
    """
    Implements Chain of Thought prompting for LLMs.
    """

    def __init__(self, llm: LLMInterface, tool: WikipediaTool):
        """
        Initialize the ChainOfThought instance.

        Args:
            llm (LLMInterface): An instance of LLMInterface.
            tool (WikipediaTool): An instance of WikipediaTool.
        """
        self.llm = llm
        self.tool = tool

    def generate_response(self, exemplar: str, question: str) -> str:
        """
        Generate a response using Chain of Thought prompting.

        Args:
            exemplar (str): The exemplar demonstrating reasoning steps.
            question (str): The question to be answered.

        Returns:
            str: The LLM's response.
        """
        llm_call = f"{exemplar}{question}\nA:"
        response = self.llm.call_llm(llm_call)
        return response

