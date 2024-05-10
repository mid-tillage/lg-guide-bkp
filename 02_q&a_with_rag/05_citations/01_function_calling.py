from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import dotenv

dotenv.load_dotenv()


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
wiki = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, answer the user question. If none of the articles answer the question, just say you don't know.\n\nHere are the Wikipedia articles:{context}",
        ),
        ("human", "{question}"),
    ]
)
prompt.pretty_print()

def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


format = itemgetter("docs") | RunnableLambda(format_docs)
# subchain for generating an answer once we've done retrieval
answer = prompt | llm | StrOutputParser()
# complete chain that calls wiki -> formats docs to string -> runs answer subchain -> returns just the answer and retrieved docs.
chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=wiki)
    .assign(context=format)
    .assign(answer=answer)
    .pick(["answer", "docs"])
)

# question = "How fast are cheetahs?"
# result = chain.invoke(question)

# print("Question: " + question)
# print("Answer:" + str(result))

from langchain_core.pydantic_v1 import BaseModel, Field


class cited_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )

llm_with_tool = llm.bind_tools(
    [cited_answer],
    tool_choice="cited_answer",
)
example_q = """What Brian's height?

Source: 1
Information: Suzy is 6'2"

Source: 2
Information: Jeremiah is blonde

Source: 3
Information: Brian is 3 inches shorted than Suzy"""
llm_with_tool.invoke(example_q)

from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

output_parser = JsonOutputKeyToolsParser(key_name="cited_answer", first_tool_only=True)
(llm_with_tool | output_parser).invoke(example_q)

def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


format_1 = itemgetter("docs") | RunnableLambda(format_docs_with_id)
answer_1 = prompt | llm_with_tool | output_parser
chain_1 = (
    RunnableParallel(question=RunnablePassthrough(), docs=wiki)
    .assign(context=format_1)
    .assign(cited_answer=answer_1)
    .pick(["cited_answer", "docs"])
)

question = "How fast are cheetahs?"
result = chain_1.invoke(question)

print("Question: " + question)
print("Answer:" + str(result))

class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class quoted_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )

output_parser_2 = JsonOutputKeyToolsParser(
    key_name="quoted_answer", first_tool_only=True
)
llm_with_tool_2 = llm.bind_tools(
    [quoted_answer],
    tool_choice="quoted_answer",
)
format_2 = itemgetter("docs") | RunnableLambda(format_docs_with_id)
answer_2 = prompt | llm_with_tool_2 | output_parser_2
chain_2 = (
    RunnableParallel(question=RunnablePassthrough(), docs=wiki)
    .assign(context=format_2)
    .assign(quoted_answer=answer_2)
    .pick(["quoted_answer", "docs"])
)

result_2 = chain_2.invoke(question)

print("Question: " + question)
print("Answer:" + str(result_2))