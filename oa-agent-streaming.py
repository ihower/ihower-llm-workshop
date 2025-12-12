from agents import Agent,  ModelSettings, Runner, function_tool

@function_tool
def generate_number() -> int:
    return 42

agent = Agent(
    name="foobar",
    model="gpt-5.2",
    instructions="""First think life and call the `generate_number` tool, 
    then just tell me the number and life of meaning about the number. No Tool Preamble. 用中文輸出""",
    tools=[generate_number],
    model_settings=ModelSettings( reasoning={ "effort": "none" }),
    #model_settings=ModelSettings( reasoning={ "effort": "low" }),
    #model_settings=ModelSettings( reasoning={ "effort": "low", "summary": "auto" }),
)

async def main():
    print("-------Streaming Start-------")
    result = Runner.run_streamed(agent, "Hello")

    async for event in result.stream_events():
        if event.type == "raw_response_event": # 這個是 Responses API 的原始 event
            # print(event)

            item = getattr(event.data, "item", None)
            if item:
                print(f"{event.data.type} ItemType: {item.type}")
            elif event.data.type in ["response.output_text.delta", "response.reasoning_summary_text.delta"]:
                print(f"{event.data.type}: {event.data.delta}")
            else:
                print(f"{event.data.type}" )
        elif event.type == "agent_updated_stream_event": # 這個是 Agents SDK 額外加的事件
            pass

    print("-------Streaming End-------")

    print(f"AI: {result.final_output}")

    chat_items = result.to_input_list()
    print("---chat history---")
    for x in chat_items:
        print(x)



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())