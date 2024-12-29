from phi.agent import Agent, RunResponse
from phi.model.ollama import Ollama

from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo


#web search agent
web_search_agent= Agent(
    name= "Web search agent",
    role ="Search the web for the informantion",
    model =Ollama(id="llama3.2"),
    tools = [DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown= True
)

#financial agent
finance_agent = Agent(
    name="Finance Agent",
    model = Ollama(id="llama3.2"),
    tools =[
        YFinanceTools(stock_price= True, analyst_recommendations= True,stock_fundamentals= True, company_news = True),

    ],
    instructions=["use tables to display data"],
    show_tool_calls=True,
    markdown= True,
)

multi_ai_agent= Agent(
    model = Ollama(id="llama3.2"),
    team = [web_search_agent, finance_agent],
    instructions=["Always include sources","use tables to display data"],

)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NV",stream=True)

