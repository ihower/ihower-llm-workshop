# https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
import tiktoken

def num_tokens_from_messages(messages, model="gpt-5"):
    encoding = tiktoken.encoding_for_model(model)

    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            # print(f"  key: {key}, value: {value} type: {type(value)}")

            if key == "content":
                if isinstance(value, list):
                    for v2 in value:
                        for k3, v3 in v2.items():
                            if k3 == "text":
                                num_tokens += len(encoding.encode(str(v3)))
                            elif k3 == 'annotations': # 內建的 web search
                                for v4 in v3:
                                    for k5, v5 in v4.items():
                                        if k5 == "title" or k5 == "url":
                                            num_tokens += len(encoding.encode(str(v5)))
                else:
                    num_tokens += len(encoding.encode(str(value)))
            elif key in ("output", "arguments", "role", "action", "type"):
                num_tokens += len(encoding.encode(str(value)))
            else:
                pass

            num_tokens += tokens_per_name # for each key

    num_tokens += 3
    return num_tokens

def num_tokens_for_tools(functions, messages, model="gpt-5"):
    # Set function settings
    func_init = 7
    prop_init = 3
    prop_key = 3
    enum_init = -3
    enum_item = 3
    func_end = 12

    encoding = tiktoken.encoding_for_model(model)

    func_token_count = 0
    if len(functions) > 0:
        for x in functions:
            if x["type"] != "function": # 不支援內建工具的計算
                continue
            
            f = {
               "type": "function",
               "function": x
            }

            func_token_count += func_init  # Add tokens for start of each function
            function = f["function"]
            f_name = function["name"]
            f_desc = function.get("description", "")

            if f_desc.endswith("."):
                f_desc = f_desc[:-1]
            line = f_name + ":" + f_desc
            func_token_count += len(encoding.encode(line))  # Add tokens for set name and description
            if len(function["parameters"]["properties"]) > 0:
                func_token_count += prop_init  # Add tokens for start of each property
                for key in list(function["parameters"]["properties"].keys()):
                    func_token_count += prop_key  # Add tokens for each set property
                    p_name = key
                    p_type = function["parameters"]["properties"][key]["type"]
                    p_desc = function["parameters"]["properties"][key].get("description", "")
                    if "enum" in function["parameters"]["properties"][key].keys():
                        func_token_count += enum_init  # Add tokens if property has enum list
                        for item in function["parameters"]["properties"][key]["enum"]:
                            func_token_count += enum_item
                            func_token_count += len(encoding.encode(item))
                    if p_desc.endswith("."):
                        p_desc = p_desc[:-1]
                    line = f"{p_name}:{p_type}:{p_desc}"
                    func_token_count += len(encoding.encode(line))
        func_token_count += func_end

    messages_token_count = num_tokens_from_messages(messages, model)
    total_tokens = messages_token_count + func_token_count

    return total_tokens

from agents import RunContextWrapper
from agents.models.openai_responses import Converter

async def num_tokens_for_agent_input_items(agent, messages, model="gpt-5"):
    ctx = RunContextWrapper(context=None)
    tools = await agent.get_all_tools(ctx)

    converted = Converter.convert_tools(tools, agent.handoffs).tools

    #print(f"converted: {converted}")

    return num_tokens_for_tools(converted, messages)

def count_tokens(text: str, model: str = "gpt-5") -> int:
    """Count tokens in a text string using tiktoken encoding for the specified model."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Test real usage
if __name__ == "__main__":  
    import asyncio

    async def main():

        from openai import OpenAI

        from dotenv import load_dotenv
        load_dotenv(".env", override=True)


        client = OpenAI()

        from agents import Agent, function_tool, Runner, WebSearchTool, ModelSettings
        
        @function_tool # (description_override="Call this tool to search the web")
        async def web_search(query: str) -> str:
            """Call this tool to search the web"""

            return "web_search 1 test"

        @function_tool # (description_override="Call this tool to search the web")
        async def web_search2(query: str) -> str:
            """Call this tool to search the web hi"""

            return "web_search 2 2"

        agent = Agent(
            name="Test Agent",
        # instructions= f"hi",
            tools=[web_search],
            model="gpt-4.1-mini",
            #model_settings=ModelSettings(
            #      include=["web_search_call.action.sources"],
            #)
        )

        print("--------------Round 1--------------")
        messages = [
            { "role": "developer", "content": "hi" },
            { "role": "user", "content": "call web_search for ihower, then search AIHAO.tw" },
        # { "role": "assistant", "content": "fine" },
        # { "role": "user", "content": "yo" },
        ]

        print(f"num_tokens_from_messages: {num_tokens_from_messages(messages)}")

        print(await num_tokens_for_agent_items(agent, messages))

        print("-------")
        result = await Runner.run(agent, input=messages)
        # print(new_inputs)

        print(result.context_wrapper.usage) # 這是兩個 API call 的 input_tokens 總和, 所以比上面的數字多

        print("--------------Round 2--------------")
        
        messages = result.to_input_list()
        messages.append(
            { "role": "user", "content": "yo" },
        )
        result = await Runner.run(agent, input=messages)
      
        print(f"num_tokens_from_messages: {num_tokens_from_messages(messages)}")
        print(await num_tokens_for_agent_items(agent, messages))

        print(result.context_wrapper.usage)

    asyncio.run(main())