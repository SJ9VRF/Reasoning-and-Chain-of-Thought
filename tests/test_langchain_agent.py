# additional scenarios for the LangchainReActAgent class, including handling multiple questions and agent errors.
# tests/test_langchain_agent.py

import unittest
from unittest.mock import MagicMock, patch
from src.langchain_agent import LangchainReActAgent

class TestLangchainReActAgentAdditional(unittest.TestCase):
    @patch('langchain.agents.initialize_agent')
    @patch('langchain.tools.load_tools')
    @patch('langchain.llms.VertexAI')
    def test_run_query_with_different_questions(self, mock_vertexai, mock_load_tools, mock_initialize_agent):
        mock_llm_instance = MagicMock()
        mock_vertexai.return_value = mock_llm_instance
        mock_tool = MagicMock()
        mock_load_tools.return_value = [mock_tool]
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent

        agent = LangchainReActAgent(model_name="text-bison@001", project_id="test_project", location="us-central1")

        # Test question 1
        mock_agent.run.return_value = "Olaf Scholz"
        response1 = agent.run_query("Who is Chancellor of Germany?")
        self.assertEqual(response1, "Olaf Scholz")
        mock_agent.run.assert_called_with("Who is Chancellor of Germany?")

        # Test question 2
        mock_agent.run.return_value = "Python is a programming language."
        response2 = agent.run_query("What is Python?")
        self.assertEqual(response2, "Python is a programming language.")
        mock_agent.run.assert_called_with("What is Python?")

    @patch('langchain.agents.initialize_agent')
    @patch('langchain.tools.load_tools')
    @patch('langchain.llms.VertexAI')
    def test_run_query_with_verbose(self, mock_vertexai, mock_load_tools, mock_initialize_agent):
        mock_llm_instance = MagicMock()
        mock_vertexai.return_value = mock_llm_instance
        mock_tool = MagicMock()
        mock_load_tools.return_value = [mock_tool]
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent

        agent = LangchainReActAgent(model_name="text-bison@001", project_id="test_project", location="us-central1")

        with patch('builtins.print') as mock_print:
            mock_agent.run.return_value = "Olaf Scholz"
            response = agent.run_query("Who is Chancellor of Germany?", verbose=True)
            mock_agent.run.assert_called_with("Who is Chancellor of Germany?")
            self.assertEqual(response, "Olaf Scholz")
            mock_print.assert_not_called()  # Since verbose=True should not print anything here

    @patch('langchain.agents.initialize_agent')
    @patch('langchain.tools.load_tools')
    @patch('langchain.llms.VertexAI')
    def test_run_query_with_agent_error(self, mock_vertexai, mock_load_tools, mock_initialize_agent):
        mock_llm_instance = MagicMock()
        mock_vertexai.return_value = mock_llm_instance
        mock_tool = MagicMock()
        mock_load_tools.return_value = [mock_tool]
        mock_agent = MagicMock()
        mock_agent.run.side_effect = Exception("Agent error occurred.")
        mock_initialize_agent.return_value = mock_agent

        agent = LangchainReActAgent(model_name="text-bison@001", project_id="test_project", location="us-central1")

        with self.assertRaises(Exception) as context:
            agent.run_query("Who is Chancellor of Germany?")
        
        self.assertTrue("Agent error occurred." in str(context.exception))

if __name__ == '__main__':
    unittest.main()
