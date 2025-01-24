# ensuring SelfConsistency class correctly aggregates multiple LLM responses and plots the distribution.

# tests/test_self_consistency.py

import unittest
from unittest.mock import MagicMock, patch
from collections import Counter
from src.self_consistency import SelfConsistency
from src.llm_interface import LLMInterface
from src.tools import WikipediaTool

class TestSelfConsistency(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock(spec=LLMInterface)
        self.tool = MagicMock(spec=WikipediaTool)
        self.self_consistency = SelfConsistency(llm=self.llm, tool=self.tool)

    def test_run_multiple_responses(self):
        prompt = "What is the capital of France?"
        parameters = {"temperature": 0.7, "max_output_tokens": 100, "top_p": 1, "top_k": 40}
        responses = [
            "The answer is Paris.",
            "The capital of France is Paris.",
            "Paris is the capital of France.",
            "The answer is Paris.",
            "Paris is the capital of France.",
            "Paris is the capital.",
            "Paris is the capital of France.",
            "Paris is the capital of France.",
            "The answer is Paris.",
            "Paris is the capital of France."
        ]
        self.llm.call_llm.side_effect = responses

        with patch('builtins.print') as mock_print:
            answer_counts = self.self_consistency.run_multiple_responses(prompt, parameters, runs=10)
            expected_counter = Counter({
                "Paris": 3,
                "Paris is the capital of France": 6,
                "NA": 1
            })
            self.assertEqual(answer_counts, expected_counter)

    def test_plot_answer_distribution(self):
        # Since plotting is a visual output, we'll ensure no exceptions are raised
        answer_counts = Counter({
            "Paris": 4,
            "Paris is the capital of France": 5,
            "NA": 1
        })
        try:
            self.self_consistency.plot_answer_distribution(answer_counts)
        except Exception as e:
            self.fail(f"plot_answer_distribution raised an exception {e}")

    def test_run_multiple_responses_with_varied_answers(self):
        prompt = "What is the capital of Germany?"
        parameters = {"temperature": 0.7, "max_output_tokens": 100, "top_p": 1, "top_k": 40}
        responses = [
            "The answer is Berlin.",
            "The capital of Germany is Berlin.",
            "Berlin is the capital of Germany.",
            "The answer is Berlin.",
            "Berlin is the capital of Germany.",
            "Berlin is the capital.",
            "Berlin is the capital of Germany.",
            "Berlin is the capital of Germany.",
            "The answer is Berlin.",
            "Berlin is the capital of Germany."
        ]
        self.llm.call_llm.side_effect = responses

        with patch('builtins.print') as mock_print:
            answer_counts = self.self_consistency.run_multiple_responses(prompt, parameters, runs=10)
            expected_counter = Counter({
                "Berlin": 3,
                "Berlin is the capital of Germany": 6,
                "NA": 1
            })
            self.assertEqual(answer_counts, expected_counter)

    def test_run_multiple_responses_with_all_na(self):
        prompt = "What is the capital of Germany?"
        parameters = {"temperature": 0.7, "max_output_tokens": 100, "top_p": 1, "top_k": 40}
        responses = [
            "I don't know.",
            "Cannot determine.",
            "Not sure.",
            "No answer.",
            "NA",
            "Still unsure.",
            "No info.",
            "No answer provided.",
            "NA",
            "Cannot find the answer."
        ]
        self.llm.call_llm.side_effect = responses

        with patch('builtins.print') as mock_print:
            answer_counts = self.self_consistency.run_multiple_responses(prompt, parameters, runs=10)
            expected_counter = Counter({
                "NA": 2
            })
            self.assertEqual(answer_counts, expected_counter)

if __name__ == '__main__':
    unittest.main()

