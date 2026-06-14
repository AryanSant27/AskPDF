from Agent.state import AgentState
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

def web_search_approval_node(state: AgentState):
    logs = list(state.get("logs", []))
    logs.append("Web search queries approved/modified by user.")
    return {"logs": logs}

def run_web_search_node(state: AgentState):
    queries = state.get("web_queries", [])
    logs = list(state.get("logs", []))
    scraped_data = list(state.get("scraped_data", []))
    
    if not queries:
        logs.append("No web search queries to run.")
        return {"logs": logs}
        
    logs.append(f"Running web search for {len(queries)} queries...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for q in queries:
        logs.append(f"Web search for: '{q}'")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=2))
            
            for res in results:
                url = res.get("href")
                title = res.get("title", "Web Page")
                snippet = res.get("body", "")
                
                if not url:
                    continue
                
                if any(item["url"] == url for item in scraped_data):
                    continue
                    
                logs.append(f"Scraping content from: {url}")
                try:
                    response = requests.get(url, headers=headers, timeout=8)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        for tag in soup(["script", "style", "nav", "header", "footer"]):
                            tag.decompose()
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        cleaned_text = "\n".join(chunk for chunk in chunks if chunk)
                        
                        scraped_data.append({
                            "title": title,
                            "url": url,
                            "text": cleaned_text[:2500],
                            "snippet": snippet
                        })
                        logs.append(f"Successfully scraped '{title}' ({len(cleaned_text[:2500])} chars)")
                    else:
                        logs.append(f"Failed to scrape {url}. Status: {response.status_code}")
                        scraped_data.append({
                            "title": title,
                            "url": url,
                            "text": snippet,
                            "snippet": snippet
                        })
                except Exception as e:
                    print(f"Error scraping url {url}: {e}")
                    logs.append(f"Error scraping {url}. Falling back to snippet.")
                    scraped_data.append({
                        "title": title,
                        "url": url,
                        "text": snippet,
                        "snippet": snippet
                    })
        except Exception as e:
            print(f"Web search query '{q}' failed: {e}")
            logs.append(f"Web search failed for query: '{q}'")
            
    return {
        "scraped_data": scraped_data,
        "logs": logs
    }
