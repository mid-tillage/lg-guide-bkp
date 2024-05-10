import dotenv

dotenv.load_dotenv()

from langchain_core.tools import tool

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

from langchain.tools.render import render_text_description

rendered_tools = render_text_description([multiply])

from langchain_core.prompts import ChatPromptTemplate

system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# chain = prompt | model | JsonOutputParser()
# print(chain.invoke({"input": "what's thirteen times 4"}))

from operator import itemgetter

chain = prompt | model | JsonOutputParser() | itemgetter("arguments") | multiply
print(chain.invoke({"input": "what's thirteen times 4"}))