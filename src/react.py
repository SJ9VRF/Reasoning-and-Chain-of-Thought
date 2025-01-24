# Implements ReAct prompting.

# src/react.py

from llm_interface import LLMInterface
from tools import WikipediaTool

class ReAct:
    """
    Implements ReAct (Reasoning + Acting) prompting for LLMs.
    """

    def __init__(self, llm: LLMInterface, tool: WikipediaTool):
        """
        Initialize the ReAct instance.

        Args:
            llm (LLMInterface): An instance of LLMInterface.
            tool (WikipediaTool): An instance of WikipediaTool.
        """
        self.llm = llm
        self.tool = tool

    def get_wiki_query(self, llm_response: str, stop_text: str = "<STOP>") -> str:
        """
        Extract the Wikipedia query from the LLM's response.

        Args:
            llm_response (str): The LLM's response containing the action.
            stop_text (str): The delimiter indicating the end of the action.

        Returns:
            str: The extracted Wikipedia query.
        """
        first_line = llm_response.splitlines()[0]
        query = first_line.split(stop_text)[0]
        return query.strip()

    def react_chain(self, context: str, exemplar: str, question: str, max_steps: int = 7, show_activity: bool = False) -> str:
        """
        Execute a ReAct chain to answer a question.

        Args:
            context (str): Instructions for the LLM.
            exemplar (str): An exemplar demonstrating ReAct steps.
            question (str): The question to be answered.
            max_steps (int): Maximum number of ReAct steps.
            show_activity (bool): Whether to print activity logs.

        Returns:
            str: The final answer from the LLM.
        """
        next_llm_call = f"{context}\n\n{exemplar}\n\nQuestion: {question}\nThought 1:"
        step = 1

        while step <= max_steps:
            if show_activity:
                print(f"\033[1mReAct chain step {step}:\033[0m\x1B[0m")
            llm_response = self.llm.call_llm(next_llm_call, show_activity)

            # Check for an answer
            response_first_line = llm_response.splitlines()[0]
            first_line_answer_split = response_first_line.split("Answer[")
            if len(first_line_answer_split) > 1:
                return first_line_answer_split[1].split("]")[0]

            # Assume the second line is the action
            if len(llm_response.splitlines()) < 2:
                break  # Incomplete response
            response_second_line = llm_response.splitlines()[1]
            wiki_query = self.get_wiki_query(response_second_line)
            wiki_text = self.tool.wiki_tool(wiki_query)

            # Assemble the next LLM call
            usable_response = f"{response_first_line}\n{response_second_line}"
            obs = f"Observation {step}: {wiki_text}"
            step += 1
            next_llm_call = f"{next_llm_call} {usable_response}\n{obs}\nThought {step}:"

        return None  # Max steps exceeded without finding an answer


