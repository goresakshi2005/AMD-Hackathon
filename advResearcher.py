import requests
import wikipedia
from ddgs import DDGS
from tavily import TavilyClient
from exa_py import Exa
from openai import OpenAI
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import pickle
from pathlib import Path
import re

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


@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    content: str
    url: str
    source: str
    full_content: str = ""  # Store full content separately
    relevance_score: float = 1.0
    timestamp: str = datetime.now().isoformat()


class AdvancedResearcher:
    def __init__(self):
        self.cache_dir = Path("search_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, query: str, source: str) -> str:
        """Generate cache key for query"""
        combined = f"{source}:{query.lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Retrieve cached result if exists and not expired (24 hours)"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Check if cache is less than 24 hours old
                    if (datetime.now() - data['timestamp']).seconds < 86400:
                        return data['content']
            except:
                pass
        return None
    
    def _cache_result(self, cache_key: str, content: List[SearchResult]):
        """Cache search result"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'content': content,
                    'timestamp': datetime.now()
                }, f)
        except:
            pass
    
    def tavily_search(self, query: str) -> List[SearchResult]:
        """Tavily search with full content"""
        cache_key = self._get_cache_key(query, 'tavily')
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        results = []
        try:
            tavily_response = tavily.search(
                query=query,
                search_depth="advanced",
                max_results=10,
                include_answer=True,
                include_raw_content=True
            )
            
            for result in tavily_response.get("results", [])[:10]:
                # Get full content
                content = result.get('content', '')
                raw_content = result.get('raw_content', '')
                full_content = raw_content if raw_content else content
                
                search_result = SearchResult(
                    title=result.get('title', 'No title'),
                    content=content[:500] + "..." if len(content) > 500 else content,  # Preview
                    full_content=full_content,  # Store full content
                    url=result.get('url', ''),
                    source='Tavily',
                    relevance_score=result.get('score', 0.5)
                )
                results.append(search_result)
            
            self._cache_result(cache_key, results)
            
        except Exception as e:
            print(f"  ⚠️ Tavily search error: {str(e)[:50]}...")
        
        return results
    
    def exa_search(self, query: str) -> List[SearchResult]:
        """Exa search with full content"""
        cache_key = self._get_cache_key(query, 'exa')
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        results = []
        try:
            exa = Exa(api_key=EXA_API_KEY)
            
            search_response = exa.search(
                query,
                num_results=10
            )
            
            if hasattr(search_response, 'results'):
                for r in search_response.results[:10]:
                    # Get content
                    content = ""
                    full_content = ""
                    
                    if hasattr(r, 'text'):
                        content = r.text
                        full_content = r.text
                    elif hasattr(r, 'content'):
                        content = r.content
                        full_content = r.content
                    elif hasattr(r, 'snippet'):
                        content = r.snippet
                        full_content = r.snippet
                    
                    search_result = SearchResult(
                        title=getattr(r, 'title', 'No title') or 'No title',
                        content=content[:500] + "..." if content and len(content) > 500 else (content or ""),
                        full_content=full_content or "",
                        url=getattr(r, 'url', ''),
                        source='Exa',
                        relevance_score=0.8
                    )
                    results.append(search_result)
            
        except Exception as e:
            print(f"  ⚠️ Exa search error: {str(e)[:50]}...")
        
        self._cache_result(cache_key, results)
        return results
    
    def serper_search(self, query: str) -> List[SearchResult]:
        """Serper search"""
        cache_key = self._get_cache_key(query, 'serper')
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        results = []
        try:
            url = "https://google.serper.dev/search"
            payload = {
                "q": query,
                "num": 10
            }
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process organic results
            for r in data.get("organic", [])[:10]:
                search_result = SearchResult(
                    title=r.get('title', 'No title'),
                    content=r.get('snippet', ''),
                    full_content=r.get('snippet', '') + "\n\n" + r.get('description', '') if r.get('description') else r.get('snippet', ''),
                    url=r.get('link', ''),
                    source='Google',
                    relevance_score=1.0 - (r.get('position', 1) - 1) * 0.1
                )
                results.append(search_result)
            
        except Exception as e:
            print(f"  ⚠️ Serper search error: {str(e)[:50]}...")
        
        self._cache_result(cache_key, results)
        return results
    
    def duckduckgo_search(self, query: str) -> List[SearchResult]:
        """DuckDuckGo search"""
        cache_key = self._get_cache_key(query, 'duckduckgo')
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=10):
                    search_result = SearchResult(
                        title=r.get('title', 'No title'),
                        content=r.get('body', '')[:500] + "..." if r.get('body') and len(r.get('body', '')) > 500 else r.get('body', ''),
                        full_content=r.get('body', ''),
                        url=r.get('href', ''),
                        source='DuckDuckGo',
                        relevance_score=0.8
                    )
                    results.append(search_result)
                    
        except Exception as e:
            print(f"  ⚠️ DuckDuckGo search error: {str(e)[:50]}...")
        
        self._cache_result(cache_key, results)
        return results
    
    def wikipedia_search(self, query: str) -> List[SearchResult]:
        """Wikipedia search"""
        cache_key = self._get_cache_key(query, 'wikipedia')
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        results = []
        try:
            wikipedia.set_lang('en')
            search_results = wikipedia.search(query, results=5)
            
            for title in search_results[:5]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    search_result = SearchResult(
                        title=page.title,
                        content=page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                        full_content=page.summary + "\n\n" + page.content[:2000] if hasattr(page, 'content') else page.summary,
                        url=page.url,
                        source='Wikipedia',
                        relevance_score=0.9
                    )
                    results.append(search_result)
                except:
                    continue
                    
        except Exception as e:
            print(f"  ⚠️ Wikipedia search error: {str(e)[:50]}...")
        
        self._cache_result(cache_key, results)
        return results
    
    def deduplicate_results(self, all_results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results"""
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        for result in all_results:
            # Skip if URL already seen
            if result.url and result.url in seen_urls:
                continue
            
            # Skip if very similar title
            title_lower = result.title.lower()
            if any(self._title_similarity(title_lower, seen) > 0.8 for seen in seen_titles):
                continue
            
            if result.url:
                seen_urls.add(result.url)
            seen_titles.add(title_lower)
            unique_results.append(result)
            
            if len(unique_results) >= 20:
                break
        
        return unique_results
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Simple title similarity check"""
        if not title1 or not title2:
            return 0.0
        
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def synthesize_results(self, query: str, results: List[SearchResult]) -> str:
        """Synthesize results with full content"""
        
        if not results:
            return f"# Research Results: {query}\n\nNo results found for this query."
        
        # Prepare comprehensive content for synthesis
        comprehensive_results = ""
        for i, r in enumerate(results[:8], 1):
            comprehensive_results += f"""
--- SOURCE {i}: {r.source} ---
TITLE: {r.title}
URL: {r.url}

FULL CONTENT:
{r.full_content[:2000]}  # Send more content for better synthesis

"""
        
        prompt = f"""
You are a research archivist, NOT a summarizer.

Your job is to preserve and present ALL information from the search results.

Query: {query}

Search Results:
{comprehensive_results}

CRITICAL INSTRUCTIONS:

- DO NOT summarize
- DO NOT shorten
- DO NOT compress information
- DO NOT remove details
- DO NOT merge points that lose information
- DO NOT rewrite in shorter form

Your job is ONLY to:

1. Extract ALL important content from EACH source
2. Preserve original meaning fully
3. Present it in structured Markdown
4. Clearly separate each source
5. Include ALL examples, explanations, and evidence
6. Include ALL statistics, quotes, and technical details

FORMAT:

# Research Results for: {query}

---

# Source 1: [Source Name](URL)

## Full Extracted Content
(all information from this source)

---

# Source 2: [Source Name](URL)

## Full Extracted Content
(all information from this source)

---

Repeat for ALL sources.

---

# Cross-Source Insights

Only list relationships if they exist.

DO NOT summarize.

---

# Final Notes

Mention missing information if any.

IMPORTANT:
This is a FULL research archive, NOT a summary.
Nothing important should be lost.
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4.5-mini",  # Use 16k model for longer context
                messages=[
                    {"role": "system", "content": "You are an expert researcher providing comprehensive, well-structured answers with citations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            synthesized = response.choices[0].message.content
            
        except Exception as e:
            print(f"  ⚠️ OpenAI synthesis error: {str(e)[:50]}...")
            synthesized = self._comprehensive_format_results(query, results)
        
        # Add detailed sources section
        sources_section = self._format_detailed_sources(results)
        
        return synthesized + "\n\n" + sources_section
    
    def _comprehensive_format_results(self, query: str, results: List[SearchResult]) -> str:
        """Comprehensive formatting with full content"""
        output = f"# Research Results: {query}\n\n"
        output += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        # Group by source
        sources = {}
        for r in results:
            if r.source not in sources:
                sources[r.source] = []
            sources[r.source].append(r)
        
        for source_name, source_results in sources.items():
            output += f"## 📚 {source_name} Results\n\n"
            
            for i, r in enumerate(source_results[:5], 1):
                output += f"### {i}. {r.title}\n\n"
                output += f"**URL:** {r.url}\n\n"
                output += "**Content:**\n\n"
                output += r.full_content + "\n\n"
                output += "---\n\n"
        
        return output
    
    def _format_detailed_sources(self, results: List[SearchResult]) -> str:
        """Format detailed sources section"""
        output = "\n\n---\n"
        output += "## 📚 Detailed Sources\n\n"
        
        for i, r in enumerate(results[:15], 1):
            output += f"### {i}. [{r.title}]({r.url})\n"
            output += f"- **Source:** {r.source}\n"
            output += f"- **Relevance:** {r.relevance_score:.2f}\n"
            output += f"- **Preview:** {r.content[:200]}...\n\n"
        
        return output
    
    def research(self, query: str) -> Dict:
        """Main research method"""
        print(f"\n🔍 Researching: {query}")
        print("-" * 50)
        
        # Gather results from all sources
        all_results = []
        
        print("📚 Searching Tavily...", end='', flush=True)
        tavily_results = self.tavily_search(query)
        all_results.extend(tavily_results)
        print(f" found {len(tavily_results)} results")
        
        print("📚 Searching Exa...", end='', flush=True)
        exa_results = self.exa_search(query)
        all_results.extend(exa_results)
        print(f" found {len(exa_results)} results")
        
        print("📚 Searching Google...", end='', flush=True)
        serper_results = self.serper_search(query)
        all_results.extend(serper_results)
        print(f" found {len(serper_results)} results")
        
        print("📚 Searching DuckDuckGo...", end='', flush=True)
        ddg_results = self.duckduckgo_search(query)
        all_results.extend(ddg_results)
        print(f" found {len(ddg_results)} results")
        
        print("📚 Searching Wikipedia...", end='', flush=True)
        wiki_results = self.wikipedia_search(query)
        all_results.extend(wiki_results)
        print(f" found {len(wiki_results)} results")
        
        print(f"\n✅ Found {len(all_results)} total results")
        
        # Deduplicate
        print("🔄 Deduplicating results...", end='', flush=True)
        unique_results = self.deduplicate_results(all_results)
        print(f" kept {len(unique_results)} unique results")
        
        # Synthesize
        print("🧠 Synthesizing results with full content...")
        answer = self.synthesize_results(query, unique_results)
        
        return {
            'query': query,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'num_sources': len(unique_results),
            'all_results': unique_results  # Store all results for potential later use
        }


def main():
    """Main execution function"""
    researcher = AdvancedResearcher()
    
    print("=" * 60)
    print("🔬 ADVANCED AI RESEARCH ASSISTANT")
    print("=" * 60)
    print("This tool provides comprehensive research with full content")
    print("Press Ctrl+C to exit at any time")
    print("=" * 60)
    
    while True:
        try:
            print("\n" + "=" * 60)
            query = input("📝 Enter your research query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("⚠️ Please enter a valid query.")
                continue
            
            # Perform research
            result = researcher.research(query)
            
            # Save to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_{timestamp}.md"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result['answer'])
            
            print(f"\n✅ Research complete!")
            print(f"📄 Saved to: {filename}")
            print(f"📊 Total sources: {result['num_sources']}")
            
            # Preview
            print("\n" + "=" * 60)
            print("PREVIEW (first 500 chars):")
            print("=" * 60)
            preview = result['answer'][:500] + "..." if len(result['answer']) > 500 else result['answer']
            print(preview)
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()