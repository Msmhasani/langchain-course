import os
from pprint import pprint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage


load_dotenv()

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
    return "Tokyo weather is rainy"

llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages":HumanMessage(content="How is the weather in Tokyo")
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
