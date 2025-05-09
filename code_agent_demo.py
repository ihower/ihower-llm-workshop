from dotenv import load_dotenv

load_dotenv(".env", override=True)

from smolagents import CodeAgent, ToolCallingAgent, tool
from smolagents import LiteLLMModel

import litellm
litellm.callbacks = ["braintrust"]

# 初始化模型
model = LiteLLMModel(model_id="openai/gpt-4.1", temperature=0)

# 定義工具函數
@tool
def lookup_rates(country: str) -> tuple:
    """
    查詢指定國家的匯率和稅率
    
    Args:
        country (str): 國家名稱，例如 'USA', 'Japan', 'Germany', 'India'
    """
    # 返回模擬數據 (匯率, 稅率)
    fake_data = {
        "USA": (1.0, 0.08),
        "Japan": (140.0, 0.10),
        "Germany": (0.92, 0.19),
        "India": (83.5, 0.18)
    }
    return fake_data.get(country, (0, 0))

@tool
def lookup_phone_price(model: str, country: str) -> float:
    """
    查詢指定手機型號在指定國家的基本價格
    
    Args:
        model (str): 手機型號，例如 'CodeAct 1'
        country (str): 國家名稱，例如 'USA', 'Japan', 'Germany', 'India'
    """
    # 返回模擬數據
    prices = {
        "CodeAct 1": {
            "USA": 999.0,
            "Japan": 140000.0,
            "Germany": 920.0,
            "India": 83500.0
        }
    }
    return prices.get(model, {}).get(country, 0)

@tool
def convert_and_tax(price: float, exchange_rate: float, tax_rate: float) -> float:
    """
    將價格轉換並計算稅後金額
    
    Args:
        price (float): 原始價格
        exchange_rate (float): 匯率
        tax_rate (float): 稅率
    """
    return price * (1 + tax_rate) / exchange_rate

@tool
def estimate_shipping_cost(destination_country: str) -> float:
    """
    估算運送到指定國家的運費
    
    Args:
        destination_country (str): 目的地國家
    """
    # 返回模擬數據
    shipping_costs = {
        "USA": 10.0,
        "Japan": 25.0,
        "Germany": 15.0,
        "India": 30.0
    }
    return shipping_costs.get(destination_country, 0)

@tool
def estimate_final_price(converted_price: float, shipping_cost: float) -> float:
    """
    計算包含運費的最終價格
    
    Args:
        converted_price (float): 轉換後的價格(美元)
        shipping_cost (float): 運費(美元)
    """
    return converted_price + shipping_cost

# 創建代理
code_agent = CodeAgent(
    tools=[lookup_rates, lookup_phone_price, convert_and_tax, 
           estimate_shipping_cost, estimate_final_price], 
    model=model
)
tool_calling_agent = ToolCallingAgent(
    tools=[lookup_rates, lookup_phone_price, convert_and_tax, 
           estimate_shipping_cost, estimate_final_price], 
    model=model
)

query = "請判斷購買手機 'CodeAct 1' 最具成本效益的國家，需要考慮的國家有美國、日本、德國和印度。"

result = code_agent.run(query)
#result = tool_calling_agent.run(query)

print(result)