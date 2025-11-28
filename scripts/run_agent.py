import sys
from pathlib import Path
from dotenv import load_dotenv
import time
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.basic_agent import BasicAgent
from memory_strategies.full_history import FullHistoryMemory
from utils.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

# ---------
# Tools
# ---------

# Sync tool: blocks for 2 seconds
@tool
def sync_tool(x: int) -> str:
    """synchronous tool for testing"""
    t = 4
    time.sleep(t)
    return f"sync result - {t} seconds"

# Async tool: sleeps asynchronously for 2 seconds
@tool
async def async_tool(y: int) -> str:
    """asynchronous tool for testing"""
    t = 9
    await asyncio.sleep(t)
    return f"async result - {t} seconds"

tools = [async_tool, sync_tool]

# ---------
# Config
# ---------
SYSTEM_MESSAGE = SystemMessage(content="""You are a helpful AI assistant with access to tools.

IMPORTANT INSTRUCTIONS:
1. Use tools when you need information you don't have (current data, calculations, external info)
2. Respond directly when you can answer from your knowledge
3. If a tool returns an error, acknowledge it and try a different approach
4. Be concise in your responses
5. When you have gathered all necessary information from tools, provide your final answer directly

Guidelines on when to use tools:
- Need current information? → Use appropriate tool
- Can answer from knowledge? → Respond directly
- Simple questions? → Respond directly

You can respond with or without tools - use your judgment!
"""
)
LLM_MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
MAX_LLM_CALLS_COUNT = 5
MEMORY_STRATEGY = FullHistoryMemory()

agent = BasicAgent(
    tools=tools,
    memory_strategy=MEMORY_STRATEGY,
    llm=ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=TEMPERATURE
    ),
    max_llm_calls_count=MAX_LLM_CALLS_COUNT,
    system_message=SYSTEM_MESSAGE
)

# ---------
# RUN - simulate an llm query that calls both async and sync tools
# ---------
from pprint import pprint
from typing import Any, Dict

def pretty_print_response(response: Dict[str, Any]):
    print("\n=== AGENT RESPONSE ===")
    print("\nResponse:")
    print(response.get("response", ""))

    print("\nConversation History:")
    for i, msg in enumerate(response.get("conversation_history", []), 1):
        role = msg.__class__.__name__ if hasattr(msg, "__class__") else "Message"
        content = getattr(msg, "content", str(msg))
        print(f"{i}. {role}: {content}")

    print("\nScratchpad (tool calls/results):")
    for i, msg in enumerate(response.get("scratchpad", []), 1):
        role = msg.__class__.__name__ if hasattr(msg, "__class__") else "Message"
        
        # Check if this is a tool call message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"{i}. {role} [Tool Calls]:")
            for tool_call in msg.tool_calls:
                # Handle both dict and object formats
                if isinstance(tool_call, dict):
                    name = tool_call.get("name", "unknown")
                    args = tool_call.get("args", {})
                    print(f"   - {name}: {args}")
                else:
                    print(f"   - {tool_call.function.name}: {tool_call.function.arguments}")
        # Check if this is a tool result message
        elif hasattr(msg, "tool_call_id"):
            content = getattr(msg, "content", str(msg))
            print(f"{i}. {role} [Tool ID: {msg.tool_call_id}]: {content}")
        # Regular message with content
        else:
            content = getattr(msg, "content", str(msg))
            print(f"{i}. {role}: {content}")

    print("\nStop Reason:", response.get("stop_reason", ""))
    print("Thread ID:", response.get("thread_id", ""))
    print("\nMetrics:")
    pprint(response.get("metrics", {}))
    
def test_agent():
    
    thread_id = "123"
    query = "Please use all your tools once. I want to test my implementation of async and sync tools by checking the time took to run them."
    response = agent.invoke(human_query=query, thread_id=thread_id)
    
    pretty_print_response(response)
    
if __name__ == "__main__":
    test_agent()