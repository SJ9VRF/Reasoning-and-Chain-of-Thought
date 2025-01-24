# Integrates Langchain for ReAct agents.

# src/langchain_agent.py

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import VertexAI
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

class LangchainReActAgent:
    """
    Implements a ReAct agent using Langchain.
    """

    def __init__(self, model_name: str, project_id: str, location: str):
        """
        Initialize the Langchain ReAct agent.

        Args:
            model_name (str): Name of the Vertex AI model.
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud location.
        """
        import vertexai
        vertexai.init(project=project_id, location=location)
        self.llm = VertexAI(model_name=model_name, temperature=0)
        self.tools = load_tools(["wikipedia"], llm=self.llm)
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    def run_query(self, query: str, verbose: bool = False) -> str:
        """
        Run a query using the ReAct agent.

        Args:
            query (str): The query to be answered.
            verbose (bool): Whether to enable verbose mode.

        Returns:
            str: The agent's response.
        """
        return self.agent.run(query, verbose=verbose)
