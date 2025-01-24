# tests/test_react.py

# ReAct class, ensuring it correctly processes thoughts, actions, and observations to generate answers.


# tests/test_react.py

import unittest
from unittest.mock import MagicMock, patch
from src.react import ReAct
from src.llm_interface import LLMInterface
from src.tools import WikipediaTool

class TestReAct(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock(spec=LLMInterface)
        self.tool = MagicMock(spec=WikipediaTool)
        self.react = ReAct(llm=self.llm, tool=self.tool)

    def test_get_wiki_query(self):
        response = "Action 1: Python (programming language)<STOP>\nSome additional text."
        expected_query = "Python (programming language)"
        query = self.react.get_wiki_query(response)
        self.assertEqual(query, expected_query)

    def test_react_chain_with_answer(self):
        context = "Test context."
        exemplar = "Test exemplar."
        question = "What is Python?"
        self.llm.call_llm.side_effect = [
            "Thought 1: I need to look up Python.\nAction 1: Python<STOP>",
            "Thought 2: Python is a programming language.\nAnswer[Python is a programming language.]"
        ]
        self.tool.wiki_tool.return_value = "Python is a programming language."

        answer = self.react.react_chain(context, exemplar, question, max_steps=3, show_activity=False)
        self.llm.call_llm.assert_any_call(f"{context}\n\n{exemplar}\n\nQuestion: {question}\nThought 1:", show_activity=False)
        self.tool.wiki_tool.assert_called_with("Python")
        self.llm.call_llm.assert_any_call(f"{context}\n\n{exemplar}\n\nQuestion: {question}\nThought 1: I need to look up Python.\nAction 1: Python<STOP>\nObservation 1: Python is a programming language.\nThought 2:", show_activity=False)
        self.assertEqual(answer, "Python is a programming language.")

    def test_react_chain_max_steps_exceeded(self):
        context = "Test context."
        exemplar = "Test exemplar."
        question = "What is Python?"
        # Simulate no answer within max_steps
        self.llm.call_llm.side_effect = [
            "Thought 1: I need to look up Python.\nAction 1: Python<STOP>",
            "Thought 2: Python is a programming language.\nAction 2: AnotherQuery<STOP>",
            "Thought 3: Further reasoning without answer.\nAction 3: YetAnotherQuery<STOP>"
        ]
        self.tool.wiki_tool.side_effect = [
            "Python is a programming language.",
            "Another content.",
            "Yet another content."
        ]

        answer = self.react.react_chain(context, exemplar, question, max_steps=3, show_activity=False)
        self.assertIsNone(answer)

    def test_react_chain_with_incomplete_response(self):
        context = "Test context."
        exemplar = "Test exemplar."
        question = "What is Python?"
        # Simulate incomplete response with only one line
        self.llm.call_llm.side_effect = [
            "Thought 1: I need to look up Python."
        ]
        self.tool.wiki_tool.return_value = "Python is a programming language."

        answer = self.react.react_chain(context, exemplar, question, max_steps=3, show_activity=False)
        self.assertIsNone(answer)

    def test_react_chain_with_non_standard_answer_format(self):
        context = "Test context."
        exemplar = "Test exemplar."
        question = "What is Python?"
        # Answer not in the expected format
        self.llm.call_llm.side_effect = [
            "Thought 1: I need to look up Python.\nAction 1: Python<STOP>",
            "Thought 2: Python is a programming language.\nAnswer: Python is great."
        ]
        self.tool.wiki_tool.return_value = "Python is a programming language."

        answer = self.react.react_chain(context, exemplar, question, max_steps=3, show_activity=False)
        self.assertEqual(answer, "Python is great.")

    def test_react_chain_with_multiple_actions(self):
        context = "Test context."
        exemplar = "Test exemplar."
        question = "Who developed Python?"
        self.llm.call_llm.side_effect = [
            "Thought 1: I need to look up Python.\nAction 1: Python<STOP>",
            "Thought 2: I need to find out who developed Python.\nAction 2: Guido van Rossum<STOP>",
            "Thought 3: Guido van Rossum developed Python. Answer[Guido van Rossum]"
        ]
        self.tool.wiki_tool.side_effect = [
            "Python is a programming language.",
            "Guido van Rossum is the creator of Python."
        ]

        answer = self.react.react_chain(context, exemplar, question, max_steps=3, show_activity=False)
        self.assertEqual(answer, "Guido van Rossum")

    def test_react_chain_with_looping_actions(self):
        context = "Test context."
        exemplar = "Test exemplar."
        question = "What is Python?"
        # Simulate looping by repeating the same action
        self.llm.call_llm.side_effect = [
            "Thought 1: I need to look up Python.\nAction 1: Python<STOP>",
            "Thought 2: I need to look up Python again.\nAction 2: Python<STOP>",
            "Thought 3: Still need to look up Python.\nAction 3: Python<STOP>",
            "Thought 4: No progress made.\nAction 4: Python<STOP>",
            "Thought 5: I give up.\nAnswer[Python is a programming language.]"
        ]
        self.tool.wiki_tool.return_value = "Python is a programming language."

        answer = self.react.react_chain(context, exemplar, question, max_steps=5, show_activity=False)
        self.assertEqual(answer, "Python is a programming language.")

if __name__ == '__main__':
    unittest.main()
