from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import dotenv

dotenv.load_dotenv()

# Initialize ChatOpenAI model
llm = ChatOpenAI()

# Load documents from a web-based source
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Split text into documents using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Create vector store using FAISS
vector = FAISS.from_documents(documents, embeddings)

# Define prompt template for answering questions
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retriever from vector store
retriever = vector.as_retriever()

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Invoke retrieval chain with a question
user_input = "how can langsmith help with testing?"
result = retrieval_chain.invoke({"input": user_input})

# Print the answer
print("# User input: " + user_input)
print("# Answer:" + result["answer"])
print("\n")

# Define prompt template for generating search query based on conversation history
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

# Create history-aware retriever chain
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Define chat history
user_input = "Can LangSmith help test my LLM applications?"
chat_history = [HumanMessage(content=user_input), AIMessage(content="Yes!")]

# Invoke history-aware retriever chain
result = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

# Print the length of the result and the first element
print("# User input: " + user_input)
print("# Answer: " + str(result))
print("\n")

#                                                                                                               #
# Define prompt template for answering questions based on context and conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
# Create document chain for answering questions
document_chain = create_stuff_documents_chain(llm, prompt)

# Combine history-aware retriever chain with document chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

# Invoke combined retrieval chain with question and chat history
user_input = "Tell me how"
result = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": user_input
})

# Print the result: keys, and answer
print("# User input: " + user_input)
print("# Answer (keys " + str(result.keys()) + "): " + str(result['answer']))
print("\n")
