import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tracers.log_stream import LogEntry, LogStreamCallbackHandler
import dotenv


dotenv.load_dotenv()

# Load, chunk and index the contents of the blog.
bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs_strainer},
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = (contextualize_q_prompt | llm | StrOutputParser()).with_config(
    tags=["contextualize_q_chain"]
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(context=contextualize_q_chain | retriever | format_docs)
    | qa_prompt
    | llm
)

from typing import Any, Dict, List, Optional, TypedDict

class RunState(TypedDict):
    id: str
    """ID of the run."""
    streamed_output: List[Any]
    """List of output chunks streamed by Runnable.stream()"""
    final_output: Optional[Any]
    """Final output of the run, usually the result of aggregating (`+`) streamed_output.
    Only available after the run has finished successfully."""

    logs: Dict[str, LogEntry]
    """Map of run names to sub-runs. If filters were supplied, this list will
    contain only the runs that matched the filters."""

# Needed for running async functions in Jupyter notebook:
# import nest_asyncio

# nest_asyncio.apply()

from langchain_core.messages import HumanMessage

chat_history = []

question = "What is Task Decomposition?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg])

second_question = "What are common ways of doing it?"
ct = 0

async def run_chain():
    ct = 0
    async for jsonpatch_op in rag_chain.astream_log(
        {"question": second_question, "chat_history": chat_history},
        include_tags=["contextualize_q_chain"],
    ):
        print(jsonpatch_op)
        print("\n" + "-" * 30 + "\n")
        ct += 1
        if ct > 20:
            break

import asyncio
asyncio.run(run_chain())

# # Needed for running async functions in Jupyter notebook:
# import nest_asyncio
# nest_asyncio.apply()

# async for jsonpatch_op in rag_chain.astream_log(
#     {"question": second_question, "chat_history": chat_history},
#     include_tags=["contextualize_q_chain"],
# ):
#     print(jsonpatch_op)
#     print("\n" + "-" * 30 + "\n")
#     ct += 1
#     if ct > 20:
#         break

# If we wanted to get our retrieved docs, we could filter on name "Retriever":
ct = 0
async def run_chain_include_retrievers() :
    async for jsonpatch_op in rag_chain.astream_log(
        {"question": second_question, "chat_history": chat_history},
        include_names=["Retriever"],
        with_streamed_output_list=False,
    ):
        print(jsonpatch_op)
        print("\n" + "-" * 30 + "\n")
        ct += 1
        if ct > 20:
            break

run_chain_include_retrievers()