# (Optional) Custom callback handlers for Langchain (for observability).

# src/all_chain_details.py

from langchain.callbacks.base import BaseCallbackHandler

class AllChainDetails(BaseCallbackHandler):
    """
    Custom callback handler to log all chain details for observability.
    """

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        print("\n\033[1mLLM call started:\033[0m")
        for prompt in prompts:
            print(prompt)

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="")

    def on_llm_end(self, response: str, **kwargs):
        print("\n\033[1mLLM call ended.\033[0m")

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        print("\n\033[1mChain started:\033[0m")

    def on_chain_end(self, outputs: dict, **kwargs):
        print("\n\033[1mChain ended.\033[0m")
