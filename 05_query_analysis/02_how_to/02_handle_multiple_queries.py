import dotenv
dotenv.load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

texts = ["Harrison worked at Kensho", "Ankush worked at Facebook"]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(
    texts,
    embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Search(BaseModel):
    """Search over a database of job records."""

    queries: List[str] = Field(
        ...,
        description="Distinct queries to search for",
    )

from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

output_parser = PydanticToolsParser(tools=[Search])

system = """You have the ability to issue search queries to get information to help answer user information.

If you need to look up two distinct pieces of information, you are allowed to do that!"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

output_parser = PydanticToolsParser(tools=[Search])

system = """You have the ability to issue search queries to get information to help answer user information.

If you need to look up two distinct pieces of information, you are allowed to do that!"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

print(query_analyzer.invoke("where did Harrison Work"))
print(query_analyzer.invoke("where did Harrison and ankush Work"))

from langchain_core.runnables import chain

@chain
async def custom_chain(question):
    response = await query_analyzer.ainvoke(question)
    docs = []
    for query in response.queries:
        new_docs = await retriever.ainvoke(query)
        docs.extend(new_docs)
    # You probably want to think about reranking or deduplicating documents here
    # But that is a separate topic
    return docs

async def main():
    print(await custom_chain.ainvoke("where did Harrison Work"))
    print(await custom_chain.ainvoke("where did Harrison and ankush Work"))

import asyncio
asyncio.run(main())
