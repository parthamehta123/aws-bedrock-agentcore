#### Agent - prompt - tools - llm

import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse
from langchain_nimble import NimbleSearchTool

nimble_search = NimbleSearchTool(
    k = 5,
    deep_search = True,
    api_key = os.getenv("NIMBLE_API_KEY"),
)

llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
)

@tool
def search_tool(query: str):
    """
        Search anything on google for user queries or real-time data.
        Args:
            query - user search query
    """

    response = nimble_search.run(query)
    return response

agent = create_agent(
    model = llm,
    tools = [search_tool],
    system_prompt = """You are a search agent. Answer questions from live Google results returned by search_tool.

When to search:
- Call search_tool before answering any factual, current, or specific question, including news, weather, prices, people, events, statistics, and product details.
- Skip the tool only for greetings, thanks, or a request to clarify what the user wants.
- Do not answer factual questions from memory.

How to search:
- Turn the question into a short, specific query. Include the place, date, or name when it matters.
- If the results are missing, off-topic, or incomplete, search once more with a clearer query.
- Stop once the results are enough to answer.

How to answer:
- Use only what the search results support. Do not invent facts, numbers, dates, or sources.
- Lead with the direct answer, then the supporting details.
- If the results disagree, say what they disagree on.
- If the results do not contain the answer, say you don't know."""
)
