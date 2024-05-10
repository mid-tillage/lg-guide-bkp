from langchain_core.tools import tool
import dotenv

dotenv.load_dotenv()

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

from langchain_openai.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_with_tools = llm.bind_tools([multiply])

# from operator import itemgetter
chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
print(chain.invoke("What's four times 23"))