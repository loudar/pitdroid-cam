from intentRecognition import llm, tools


def recognize_intent(input_string):
    ai_msg = llm.invoke([
        ("system", "You are a physical robot that can do stuff that is described in tools you are given."),
        ("human", input_string),
    ], tools=tools)
    return ai_msg.response_metadata["message"]["tool_calls"][0]["function"]


test_commands = [
    "Move forward",
    "Move backward",
    "Move left",
    "Move right",
    "Come towards me",
    "Walk away from me!",
    "Walk this long way in front of you!"
]

for command in test_commands:
    call = recognize_intent(command)
    print(f"Command {command} || {call['name']} with args {call['arguments']}")
