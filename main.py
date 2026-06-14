import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()

print("Initializations...")

embeddings = OpenAIEmbeddings()

# llm = ChatOpenAI(model="gpt-5.2")
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer:"""
)

def format_docs(docs):
    '''Format retrieved documents into a single string.'''
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_withput_lcel(query: str):
    '''
    Simple retrieval chain withput LCEL.
    Manually retrieves documents, formats them, and generates a response.
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support withput addiotnal code
    - Harder to compose with other chains
    - More verbose and error-prone
    '''
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


def create_retrieval_chain_with_lcel():
    '''
    create a retrieval chain using langchain expression language (lcel).
    Returns a chain that can be invoked with {"question": "..."}
    '''
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs)
            | prompt_template | llm | StrOutputParser()
        )
    return retrieval_chain



if __name__ == "__main__":
    print("Retrieving...")

    query = "what is the Pinecone in machine learning?"

    # print("\n" + "=" * 70)
    # print("impl 0 (no RAG)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # print(result_raw.content)

    # print("\n" + "=" * 70)
    # print("impl 1: no LCEL")
    # print("=" * 70)
    # result_without_lcel = retrieval_chain_withput_lcel(query)
    # print(result_without_lcel)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
