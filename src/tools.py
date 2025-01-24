# Defines external tools like Wikipedia.

# src/tools.py

import wikipedia

class WikipediaTool:
    """
    Tool for interacting with Wikipedia to fetch article snippets.
    """

    def __init__(self, return_chars: int = 1000):
        """
        Initialize the Wikipedia tool.

        Args:
            return_chars (int): Number of characters to return from the article.
        """
        self.return_chars = return_chars

    def wiki_tool(self, query: str) -> str:
        """
        Fetch a snippet from a Wikipedia article based on the query.

        Args:
            query (str): The search query for Wikipedia.

        Returns:
            str: A snippet from the Wikipedia article.
        """
        try:
            page = wikipedia.page(query, auto_suggest=False, redirect=True).content
        except wikipedia.exceptions.PageError:
            page = wikipedia.page(query, auto_suggest=True, redirect=True).content
        snippet = page[:self.return_chars]
        return snippet

