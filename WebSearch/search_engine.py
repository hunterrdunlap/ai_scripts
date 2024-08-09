import requests
from time import sleep
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import logging

def get_search_results(query: str, api_key: str, num_results: int = 5) -> List[Dict[str, Any]]:
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": num_results},
        headers=headers,
        timeout=60
    )
    if not response.ok:
        raise Exception(f"HTTP error {response.status_code}")
    sleep(1)  # avoid Brave rate limit
    return response.json().get("web", {}).get("results", [])


def get_page_content(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from the page
        text = soup.get_text(separator='\n', strip=True)
        
        # Limit the text to a reasonable length (e.g., 1000 words)
        words = text.split()
        limited_text = ' '.join(words[:1000])
        
        return limited_text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching content from {url}: {e}")
        return f"Error fetching content: {e}"
    except Exception as e:
        logging.error(f"Unexpected error processing {url}: {e}")
        return f"Error processing content: {e}"