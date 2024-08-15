from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

def get_tools():
    return [
        create_tool(
            "move",
            "Move",
            {
                "direction": {
                    "type": "integer",
                    "description": "Direction to move, as an angle from 0 to 360 degrees (0 is forward, 90 is right, 180 is backward, 270 is left)",
                    "default": "0"
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of steps to move in a given direction",
                    "default": 1
                }
            },
            ["direction", "steps"]
        ),
    ]


def create_tool(name, description, parameters, required_parameters=None):
    if required_parameters is None:
        required_parameters = []
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required_parameters
            }
        }
    }

tools = get_tools()
