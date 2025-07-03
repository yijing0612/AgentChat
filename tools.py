from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

serper = GoogleSerperAPIWrapper()
ai_search_tool = Tool(
    name="ai_web_search",
    func=serper.run,
    description="Use this for intelligent real-time web search using Google.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)