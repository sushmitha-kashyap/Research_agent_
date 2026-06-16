from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import requests
from typing import TypedDict
from newspaper import Article
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph
from langgraph.graph import END

load_dotenv()

@tool
def hackernews_top_stories(query: str) -> str:
    """
    Search HackerNews top stories.
    """

    top_stories = requests.get(
        "https://hacker-news.firebaseio.com/v0/topstories.json"
    ).json()

    results = []

    for story_id in top_stories[:10]:
        item = requests.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        ).json()

        if query.lower() in item.get("title", "").lower():
            results.append(
                {
                    "title": item["title"],
                    "url": item.get("url", ""),
                }
            )

    return str(results)


@tool
def read_article(url: str) -> str:
    """
    Read article content from URL.
    """

    article = Article(url)

    article.download()
    article.parse()

    return article.text[:5000]

@tool
def web_search(query: str) -> str:
    """
    Search web using DuckDuckGo.
    """

    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r)

    return str(results)


model = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = "conversational"
 )

llm = ChatHuggingFace(llm = model)

tools = [hackernews_top_stories,read_article,web_search]
llm_bind_tools = llm.bind_tools(tools)


hn_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a HackerNews researcher."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

hn_agent = create_tool_calling_agent(
    llm_bind_tools,
    [hackernews_top_stories],
    hn_prompt
)

hn_executor = AgentExecutor(
    agent=hn_agent,
    tools=[hackernews_top_stories]
)

reader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert article analyzer."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

reader_agent = create_tool_calling_agent(
    llm_bind_tools,
    [read_article],
    reader_prompt
)

reader_executor = AgentExecutor(
    agent=reader_agent,
    tools=[read_article]
)

search_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a web researcher."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

search_agent = create_tool_calling_agent(
    llm_bind_tools,
    [web_search],
    search_prompt
)

search_executor = AgentExecutor(
    agent=search_agent,
    tools=[web_search]
)


class ResearchState(TypedDict):
    query: str
    hn_results: str
    article_results: str
    search_results: str
    final_report: str

def hn_node(state):

    result = hn_executor.invoke({
        "input": state["query"]
    })

    return {
        "hn_results": result["output"]
    }

def article_node(state):

    result = reader_executor.invoke({
        "input": state["hn_results"]
    })

    return {
        "article_results": result["output"]
    }

def search_node(state):

    result = search_executor.invoke({
        "input": state["query"]
    })

    return {
        "search_results": result["output"]
    }

def summary_node(state):

    prompt = f"""
    User Query:
    {state['query']}

    HackerNews:
    {state['hn_results']}

    Articles:
    {state['article_results']}

    Web Search:
    {state['search_results']}

    Create a detailed report.
    """

    response = llm_bind_tools.invoke(prompt)

    return {
        "final_report": response.content
    }


builder = StateGraph(ResearchState)

builder.add_node("hn_research", hn_node)
builder.add_node("article_reader", article_node)
builder.add_node("web_search", search_node)
builder.add_node("summary", summary_node)

builder.set_entry_point("hn_research")

builder.add_edge("hn_research", "article_reader")
builder.add_edge("article_reader", "web_search")
builder.add_edge("web_search", "summary")
builder.add_edge("summary", END)

graph = builder.compile()

result = graph.invoke({
    "query": "Attention is all you need explaination"
})

print(result["final_report"])