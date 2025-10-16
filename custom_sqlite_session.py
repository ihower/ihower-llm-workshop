from agents import SQLiteSession, Agent
from agents.items import TResponseInputItem
from utils import num_tokens_for_agent_input_items, count_tokens
from pathlib import Path

# Context Engineering 閾值設定
TOOL_CALL_OUTPUT_TRIM_THRESHOLD = 150000  # 當 tokens 超過此值時，簡化 function_call_output
TURN_BASED_TRIM_THRESHOLD = 200000  # 當 tokens 超過此值時，開始移除舊的對話輪次
TURN_BASED_TARGET_TOKENS = 50000  # Turn-based trimming 的目標 token 數量

class CustomSQLiteSession(SQLiteSession):

    agent: Agent

    def __init__(
        self,
        session_id: str,
        db_path: str | Path = ":memory:",
        sessions_table: str = "agent_sessions",
        messages_table: str = "agent_messages",
        agent: Agent = None,
    ):

        super().__init__(session_id, db_path, sessions_table, messages_table)

        self.agent = agent

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:

        # Call parent's get_items to get all items
        items = await super().get_items(limit=limit)

        # 拿到 items 已經用了多少 tokens
        used_tokens = await num_tokens_for_agent_input_items(self.agent, items)
        print(f"num_tokens_for_agent_items: {used_tokens}")

        # Context Engineering 1: Tool call output trimming
        # 當 tokens 超過閾值時，簡化 function_call_output 內容
        if used_tokens > TOOL_CALL_OUTPUT_TRIM_THRESHOLD:
            print(f"Trigger tool call output filter: used_tokens={used_tokens}")
            for item in items:
                if item.get("type") == "function_call_output":
                    print(" remove function_call_output! ")
                    item["output"] = "Tool results removed (context limit). Re-run the tool if needed."

            # 重新計算 tokens
            used_tokens = await num_tokens_for_agent_input_items(self.agent, items)
            print(f"After tool call filter: {used_tokens}")

        # Context Engineering 2: Turn-based trimming
        # 當 tokens 超過閾值時，按 user turns 移除最舊的對話
        if used_tokens > TURN_BASED_TRIM_THRESHOLD:
            print(f"Trigger turn-based message filter: used_tokens={used_tokens}")

            # 按 user role 切分 turns
            turns = []  # nested list
            current_turn = []

            for item in items:
                if item.get("role") == "user" and current_turn:
                    # 遇到新的 user message，結束當前 turn
                    turns.append(current_turn)
                    current_turn = [item]
                else:
                    current_turn.append(item)

            # 添加最後一個 turn
            if current_turn:
                turns.append(current_turn)

            # 計算每個 turn 的 tokens 數量
            turn_tokens = []
            for i, turn in enumerate(turns):
                turn_content = ""
                for item in turn:
                    if item.get("content"):
                        turn_content += str(item.get("content", ""))
                    if item.get("output"):
                        turn_content += str(item.get("output", ""))

                tokens = count_tokens(turn_content)
                turn_tokens.append((i, tokens, turn))

            # 從最舊的 turn 開始移除，直到總 tokens 低於目標值
            total_tokens = sum(tokens for _, tokens, _ in turn_tokens)
            removed_turns = 0

            while total_tokens > TURN_BASED_TARGET_TOKENS and len(turn_tokens) > 1:  # 保留至少1個 turn
                # 移除最舊的 turn (索引最小的)
                removed_turn = turn_tokens.pop(0)
                total_tokens -= removed_turn[1]
                removed_turns += 1
                print(f"Removed turn {removed_turn[0]} with {removed_turn[1]} tokens")

            # 重建 items
            items = []
            for _, _, turn in turn_tokens:
                items.extend(turn)

            print(f"Token management: removed {removed_turns} turns, remaining tokens: {total_tokens}")

            # 重新計算 tokens
            used_tokens = await num_tokens_for_agent_input_items(self.agent, items)
            print(f"After turn filter: {used_tokens}")

        return items
