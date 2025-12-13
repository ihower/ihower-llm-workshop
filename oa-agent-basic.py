from agents import Agent,  ModelSettings, Runner, function_tool

@function_tool
def generate_number() -> int:
    print("generate_number called")
    return 42

agent = Agent(
    name="foobar",
    model="gpt-5.2",
    instructions="you are a helpful assistant",
    tools=[generate_number],
    #model_settings=ModelSettings( reasoning={ "effort": ""}),
    model_settings=ModelSettings( reasoning={ "effort": "low"}, response_include=["reasoning.encrypted_content"]),
)

async def main():

    print("--------------Round 1--------------")
    result1 = await Runner.run(agent, "First think and call the `generate_number` tool, then just tell me the number, only number")
    print(f"AI: {result1.final_output}")

    chat_items = result1.to_input_list()
    print("---")
    for x in chat_items:
        print(x)

    print("--------------Round 2--------------")
    chat_items.append({"role": "user", "content": "just say hello"})

    result2 = await Runner.run(agent, chat_items)
    print(f"AI: {result2.final_output}")

    chat_items = result2.to_input_list()
    print("---chat history---")
    for x in chat_items:
        print(x)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())