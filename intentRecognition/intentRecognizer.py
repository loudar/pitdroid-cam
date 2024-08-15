from intentRecognition import llm, tools


def recognize_intent(input_string, current_objects):
    print(f"Recognizing intent: {input_string}")
    ai_msg = llm.invoke([
        ("system", "You are a physical robot that can do stuff that is described in tools you are given."),
        ("system", f"You can currently see the following objects: {current_objects}"),
        ("human", input_string),
    ], tools=tools)
    call = ai_msg.response_metadata["message"]["tool_calls"][0]["function"]
    if call is None:
        print("No call found")
        return None
    print(f"f<{call['name']}>({call['arguments']})")
    return call
