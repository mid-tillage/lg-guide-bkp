from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()
result = llm.invoke("how can langsmith help with testing?")
print(result)