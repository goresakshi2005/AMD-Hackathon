import requests
import wikipedia
from ddgs import DDGS
from tavily import TavilyClient
from exa_py import Exa
from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --- Clients ---
tavily = TavilyClient(api_key=TAVILY_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------- TAVILY SEARCH ----------------

def tavily_search(query):
    tavily_response = tavily.search(
        query=query,
        search_depth="basic",
        max_results=5
    )

    content = ""
    for result in tavily_response["results"]:
        content += f"""
Title: {result['title']}
Content: {result['content']}
URL: {result['url']}

"""
    return content


# ---------------- EXA SEARCH ----------------

def exa_search(query):
    exa = Exa(EXA_API_KEY)

    results = exa.search(
        query,
        num_results=3
    )

    content = ""
    for r in results.results:
        content += f"""
Title: {r.title}
Content: {r.text}
URL: {r.url}

"""
    return content


# ---------------- SERPER SEARCH ----------------

def serper_search(query):
    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    results = response.json()["organic"]

    content = ""
    for r in results[:3]:
        content += f"""
Title: {r['title']}
Content: {r['snippet']}
URL: {r['link']}

"""
    return content


# ---------------- DUCKDUCKGO SEARCH ----------------

def duckduckgo_search(query):
    content = ""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        for r in results:
            content += f"""
Title: {r['title']}
Content: {r['body']}
URL: {r['href']}

"""
    return content


# ---------------- WIKIPEDIA SEARCH ----------------

def wikipedia_search(query):
    content = ""
    try:
        titles = wikipedia.search(query)
        for title in titles[:3]:
            page = wikipedia.page(title)
            content += f"""
Title: {title}
Content: {page.summary}
URL: {page.url}

"""
    except Exception as e:
        content += f"Error: {str(e)}\n\n"
    return content


# ---------------- COMBINED SEARCH + FORMAT ----------------

def search_and_format_md(user_query):

    # Gather results from all five sources
    tavily_results = tavily_search(user_query)
    exa_results = exa_search(user_query)
    serper_results = serper_search(user_query)
    ddg_results = duckduckgo_search(user_query)
    wiki_results = wikipedia_search(user_query)

    search_content = f"""
--- Tavily Results ---
{tavily_results}

--- Exa Results ---
{exa_results}

--- Serper Results ---
{serper_results}

--- DuckDuckGo Results ---
{ddg_results}

--- Wikipedia Results ---
{wiki_results}
"""

    # Send to OpenAI for Markdown formatting
    prompt = f"""
            You are an assistant.

            Using the search results below, answer the user query in clean Markdown format.

            User Query:
            {user_query}

            Search Results:
            {search_content}

            Return:
            - Proper Markdown
            - Headings
            - Bullet points
            - Links
        """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# --- Run ---
query = input("Enter your query: ")

answer = search_and_format_md(query)

with open("research.md", "w", encoding="utf-8") as f:
    f.write(answer)

print("\n--- Markdown Answer ---\n")
print(answer)
print("\nSaved to research.md")