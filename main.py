from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from tools import ai_search_tool, wiki_tool
import re
import json

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
memory = ConversationSummaryMemory(llm=llm, return_messages=True, memory_key="history")

instruction_block = """You are a research assistant that helps answer user questions by performing research and using tools when needed.

Always return your *final answer* as a JSON object with this format:

{{
  "topic": "Example",
  "summary": "This is a summary",
  "sources": ["source1", "source2"],
  "tools_used": ["search", "wikipedia"]
}}

Do NOT return a JSON *schema*. Only return a filled-in object.

Use appropriate tools to gather information. Summarize clearly, and cite your sources.

Never include instructions or explanation in your final output — only return the structured JSON result.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_block),
    ("placeholder", "{history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [ai_search_tool, wiki_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

def run_research_agent(query: str) -> dict:
    result = agent_executor.invoke({"query": query})
    output = result.get("output", "")

    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict) and "summary" in parsed:
            return {
                "summary": parsed.get("summary", ""),
                "sources": parsed.get("sources", []),
                "tools_used": parsed.get("tools_used", [])
            }
    except json.JSONDecodeError:
        pass

    # fallback logic ...
    return {
        "summary": output,
        "sources": [],
        "tools_used": []
    }


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

if __name__ == "__main__":
    while True:
        query = input("What can I help you research? (type 'exit' to quit): ")
        if query.lower().strip() == "exit":
            break

        raw_response = agent_executor.invoke({"query": query})
        raw_output = raw_response["output"]

        try:
            structured_response = parser.parse(raw_output)
            print(structured_response.summary)
        except Exception as e:
            print("Error parsing response:", e)
            print("Raw output:", raw_output)
