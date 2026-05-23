from typing import List

from pydantic import BaseModel, Field


import os
from pprint import pprint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer.")



@tool
def search (query: str) -> str:
    """
    A tool that search over the internet.
    Args:
        query: The query to serach for
    Returns:
        The search result
    """
    print(f"searching for {query}")
    # return "Tokyo weather is rainy"
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5")
tools = [search]
# tools = [TavilySearch()]
# agent = create_agent(model=llm, tools=tools)
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages":HumanMessage(content="Find me 1 job opening for senior computer vision engineer in Japan on Linkedin which doesn't need Japanese language and the company is not a startup.")
        }
    )
    pprint(result)
    # info = """Sasanian Empire of Persia"""
    # summary_tmp = """given the information {info} I want you to create:
    #     1. A one line summary
    #     2. two interesting facts about them
    # """
    # summary_prompt_tmplt = PromptTemplate(input_variables=["info"], template=summary_tmp)
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    # llm = ChatOllama(temperature=0, model="gemma3")
    # chain = summary_prompt_tmplt | llm
    # response = chain.invoke(input={"info": info})
    # print(response)
    # # Save response to txt file
    # with open("response.txt", "w", encoding="utf-8") as f:
    #     f.write(response.content)




if __name__ == "__main__":
    main()
