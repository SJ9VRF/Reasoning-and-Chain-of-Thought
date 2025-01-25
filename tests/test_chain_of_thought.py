# ChainOfThought class, ensuring it properly generates responses based on exemplars and questions.


import unittest
from unittest.mock import MagicMock
from src.chain_of_thought import ChainOfThought
from src.llm_interface import LLMInterface
from src.tools import WikipediaTool

class TestChainOfThought(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock(spec=LLMInterface)
        self.tool = MagicMock(spec=WikipediaTool)
        self.chain = ChainOfThought(llm=self.llm, tool=self.tool)

    def test_generate_response(self):
        exemplar = """Q: Roger has 5 tennis balls.
    He buys 2 more cans of tennis balls.
    Each can has 3 tennis balls. How many tennis balls does he have now?
    A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
    Q: """
        question = """Nomfundo writes legal briefs.
    Each brief has 3 sections, each section takes 4 hours.
    She wrote 3 briefs this week. How long did it take?"""
        prompt = f"{exemplar}{question}\nA:"
        expected_response = "It took Nomfundo 36 hours to write 3 briefs."
        self.llm.call_llm.return_value = expected_response

        response = self.chain.generate_response(exemplar, question)
        self.llm.call_llm.assert_called_with(prompt)
        self.assertEqual(response, expected_response)

    def test_generate_response_with_no_exemplars(self):
        exemplar = ""
        question = "What is the capital of France?"
        prompt = f"{exemplar}{question}\nA:"
        expected_response = "The capital of France is Paris."
        self.llm.call_llm.return_value = expected_response

        response = self.chain.generate_response(exemplar, question)
        self.llm.call_llm.assert_called_with(prompt)
        self.assertEqual(response, expected_response)

    def test_generate_response_with_large_exemplar(self):
        exemplar = """Q: What is AI?
    A: AI stands for Artificial Intelligence.
    Q: """
        question = "What is machine learning?"
        prompt = f"{exemplar}{question}\nA:"
        expected_response = "Machine learning is a subset of AI focused on building systems that learn from data."
        self.llm.call_llm.return_value = expected_response

        response = self.chain.generate_response(exemplar, question)
        self.llm.call_llm.assert_called_with(prompt)
        self.assertEqual(response, expected_response)

    def test_generate_response_tool_not_used(self):
        exemplar = """Q: What is AI?
    A: AI stands for Artificial Intelligence.
    Q: """
        question = "Define AI."
        prompt = f"{exemplar}{question}\nA:"
        expected_response = "AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines."
        self.llm.call_llm.return_value = expected_response

        response = self.chain.generate_response(exemplar, question)
        self.llm.call_llm.assert_called_with(prompt)
        self.assertEqual(response, expected_response)

if __name__ == '__main__':
    unittest.main()
