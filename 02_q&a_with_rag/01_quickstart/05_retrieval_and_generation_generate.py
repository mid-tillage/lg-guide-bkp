import getpass
import os
import dotenv
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import dotenv

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

print("len docs[0]: " + str(len(docs[0].page_content)))
print("docs[0] page content 500:" + docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print("All splits: " + str(len(all_splits)))
print("First split page content: " + str(len(all_splits[0].page_content)))
print("10° split metadata: " + str(all_splits[10].metadata))

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
print("Retrieved docs: " + str(len(retrieved_docs)))
print("Retrieved docs page content: " + retrieved_docs[0].page_content)

prompt = hub.pull("rlm/rag-prompt")

# example_messages = prompt.invoke(
#     {"context": "filler context", "question": "filler question"}
# ).to_messages()
# example_messages
# print(example_messages[0].content)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)