from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

print(multiply.invoke({"first_int": 4, "second_int": 5}))