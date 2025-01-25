# WikipediaTool class, ensuring it correctly interacts with the Wikipedia API.

# tests/test_tools.py

import unittest
from unittest.mock import patch, MagicMock
from src.tools import WikipediaTool
from wikipedia.exceptions import PageError

class TestWikipediaTool(unittest.TestCase):
    def setUp(self):
        self.tool = WikipediaTool(return_chars=1000)

    @patch('wikipedia.page')
    def test_wiki_tool_success_exact_match(self, mock_wikipedia_page):
        mock_page = MagicMock()
        mock_page.content = "This is the content of the Wikipedia page."
        mock_wikipedia_page.return_value = mock_page

        query = "Python (programming language)"
        result = self.tool.wiki_tool(query)
        mock_wikipedia_page.assert_called_with(query, auto_suggest=False, redirect=True)
        self.assertEqual(result, "This is the content of the Wikipedia page.")

    @patch('wikipedia.page')
    def test_wiki_tool_page_error(self, mock_wikipedia_page):
        # Simulate PageError on exact match and success on auto-suggest
        mock_wikipedia_page.side_effect = [PageError("Page not found"), MagicMock(content="Suggested page content.")]

        query = "NonExistentPage123"
        result = self.tool.wiki_tool(query)
        self.assertEqual(mock_wikipedia_page.call_count, 2)
        mock_wikipedia_page.assert_any_call(query, auto_suggest=False, redirect=True)
        mock_wikipedia_page.assert_any_call(query, auto_suggest=True, redirect=True)
        self.assertEqual(result, "Suggested page content.")

    @patch('wikipedia.page')
    def test_wiki_tool_snippet_length(self, mock_wikipedia_page):
        mock_page = MagicMock()
        mock_page.content = "a" * 2000  # 2000 characters
        mock_wikipedia_page.return_value = mock_page

        query = "Python (programming language)"
        result = self.tool.wiki_tool(query)
        self.assertEqual(len(result), 1000)
        self.assertEqual(result, "a" * 1000)

    @patch('wikipedia.page')
    def test_wiki_tool_empty_content(self, mock_wikipedia_page):
        mock_page = MagicMock()
        mock_page.content = ""
        mock_wikipedia_page.return_value = mock_page

        query = "EmptyPage"
        result = self.tool.wiki_tool(query)
        self.assertEqual(result, "")

    @patch('wikipedia.page')
    def test_wiki_tool_short_content(self, mock_wikipedia_page):
        mock_page = MagicMock()
        mock_page.content = "Short content."
        mock_wikipedia_page.return_value = mock_page

        query = "ShortPage"
        result = self.tool.wiki_tool(query)
        self.assertEqual(result, "Short content.")

    @patch('wikipedia.page')
    def test_wiki_tool_redirect(self, mock_wikipedia_page):
        # Simulate a redirect by having auto_suggest=False fail and auto_suggest=True succeed
        mock_wikipedia_page.side_effect = [
            PageError("Page not found"),
            MagicMock(content="Redirected page content.")
        ]

        query = "RedirectPage"
        result = self.tool.wiki_tool(query)
        self.assertEqual(mock_wikipedia_page.call_count, 2)
        self.assertEqual(result, "Redirected page content.")

if __name__ == '__main__':
    unittest.main()

