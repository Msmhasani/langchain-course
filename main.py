from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
load_dotenv()

def main():
    print("Hello from langchain-course!")
    info = """Sasanian Empire of Persia"""


    summary_tmp = """given the information {info} I want you to create:
        1. A one line summary
        2. two interesting facts about them
    """

    summary_prompt_tmplt = PromptTemplate(input_variables=["info"], template=summary_tmp)

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    # llm = ChatOllama(temperature=0, model="gemma3")
    
    chain = summary_prompt_tmplt | llm
    response = chain.invoke(input={"info": info})
    print(response)

    # Save response to txt file
    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(response.content)

if __name__ == "__main__":
    main()
