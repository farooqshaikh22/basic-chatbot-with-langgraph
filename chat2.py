import os
from dotenv import load_dotenv
from typing import List, Dict, TypedDict
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 1. Instantiate Groq LLM and Wikipedia tool
llm = ChatGroq(model="llama3-70b-8192", api_key=api_key)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki_tool.name = "wikipedia_search"
wiki_tool.description = "Search Wikipedia for a topic."

# 2. Bind the tool to the LLM
llm_with_tools = llm.bind_tools([wiki_tool], tool_choice="auto")

# 3. Create the prebuilt ReAct agent with the bound LLM
agent = create_react_agent(llm_with_tools, [wiki_tool])

# 4. Invoke and extract final answer
res = agent.invoke({"messages": [HumanMessage(content="hi")]})
print(res["messages"][-1].content)
