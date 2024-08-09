import requests
from time import sleep
from typing import List, Dict, Any

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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from the page
        text = soup.get_text(separator='\n', strip=True)
        
        # Limit the text to a reasonable length (e.g., 10000 words)
        words = text.split()
        limited_text = ' '.join(words[:10000])
        
        return limited_text
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""