import logging
import regex
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set
import csv
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time
import xml.etree.ElementTree as ET
import concurrent.futures
import threading
import queue
import json
from config import OPENAI_API_KEY, MODEL
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)


def get_base_url(url: str) -> str:
    """Extract the base URL from a given URL."""
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


def extract_emails_from_text(text: str, chunk_size: int = 100000) -> List[str]:
    """
    Extract email addresses from a given text.
    Handles various email obfuscation techniques.
    """
    # Updated email pattern to handle (dot), (at), and spaces
    email_pattern = regex.compile(
        r"\b[A-Za-z0-9._%+-]+(?:\s*(?:\(at\)|\(@\)|\[@\]|@)\s*)[A-Za-z0-9.-]+(?:\s*(?:\(dot\)|\(.\)|\[.\]|\.)\s*)[A-Za-z]{2,}\b", regex.IGNORECASE
    )
    clean_start = regex.compile(r"^[^A-Za-z0-9]+")
    clean_end = regex.compile(r"[^A-Za-z0-9._%+-]+$")
    excluded_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    valid_emails: Set[str] = set()

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        matches = email_pattern.findall(chunk)
        for match in matches:
            # Clean and normalize the email
            cleaned = clean_start.sub("", match)
            cleaned = clean_end.sub("", cleaned)
            cleaned = cleaned.replace("(at)", "@").replace("(@)", "@").replace("[@]", "@")
            cleaned = cleaned.replace("(dot)", ".").replace("(.)", ".").replace("[.]", ".")
            cleaned = "".join(cleaned.split())  # Remove all spaces

            if "@" in cleaned and "." in cleaned:
                if not any(cleaned.lower().endswith(ext) for ext in excluded_extensions):
                    valid_emails.add(cleaned)

    return list(valid_emails)


def fetch_sitemap(base_url: str) -> List[str]:
    """
    Attempt to fetch and parse the sitemap.xml file from the given base URL.
    Returns a list of URLs found in the sitemap.
    """
    sitemap_url = urljoin(base_url, "sitemap.xml")
    try:
        logger.info(f"Fetching sitemap from {sitemap_url}")
        response = requests.get(sitemap_url)
        response.raise_for_status()
        sitemap = response.content
        root = ET.fromstring(sitemap)
        urls = [elem.text for elem in root.iter() if elem.tag.endswith("loc")]
        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls
    except requests.RequestException as e:
        logger.warning(f"Error fetching sitemap: {e}")
        return []


def extract_emails_from_website(base_url: str, team_pages: List[str], max_pages: int = 10) -> List[str]:
    """
    Extract email addresses from a website by crawling its pages.
    Prioritizes team pages and respects max_pages limit.
    """
    logger.info(f"Extracting emails from website: {base_url}")
    try:
        visited = set()
        all_emails = set()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }

        # Prioritize team pages and base_url
        to_visit = team_pages + [base_url]

        website_base = get_base_url(base_url)
        base_domain = urlparse(base_url).netloc

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue

            try:
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                response.raise_for_status()

                content = ""
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    content += chunk.decode("utf-8", errors="ignore")
                    if len(content) > 10 * 1024 * 1024:
                        logger.warning(f"Page too large, stopping at 10MB: {url}")
                        break

                page_emails = extract_emails_from_text(content)
                all_emails.update(page_emails)

                soup = BeautifulSoup(content, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    full_url = urljoin(website_base, href)

                    if urlparse(full_url).netloc == base_domain:
                        if full_url not in visited and full_url not in to_visit:
                            if any(term in href.lower() for term in ["about", "contact", "über", "kontakt", "uber"]):
                                to_visit.insert(0, full_url)  # Prioritize contact pages
                            elif len(to_visit) < max_pages * 2:  # Limit size of to_visit
                                to_visit.append(full_url)

                visited.add(url)
                time.sleep(0.5)
            except requests.RequestException as e:
                logger.warning(f"Error processing {url}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error processing {url}: {str(e)}")

        logger.info(f"Found {len(all_emails)} email(s) from {len(visited)} pages")
        return list(all_emails)
    except Exception as e:
        logger.error(f"Error extracting email from website: {e}")
        return []


def ai_select_team_pages(urls: List[str], max_candidates: int = 5) -> List[str]:
    """
    Use AI to select the best candidate URLs for team pages.
    """
    logger.info(f"Using AI to select team pages from {len(urls)} URLs")
    prompt = f"""
    Please respond in this format:
    {{
        "team_pages": ["(list of the best {max_candidates} URLs for team pages)"]
    }}
    
    Select the best {max_candidates} URLs for team pages from the list provided here:
    {urls}
    
    Examples of team pages:
    - alte-post.net/team/
    - www.romantischer-winkel.de/de/hotel/unser-team.html
    - teamtegernsee.de/
    - www.burg-staufeneck.de/de/das-team
    
    Example input:
    [
        "https://example.com/about",
        "https://example.com/team",
        "https://example.com/contact",
        "https://example.com/booking",
        "https://example.com/home",
        "https://example.com/services",
        "https://example.com/pricing",
        "https://example.com/portfolio",
        "https://example.com/blog",
        "https://example.com/careers",
        "https://example.com/faq",
        "https://example.com/partners",
        "https://example.com/privacy-policy",
        "https://example.com/terms-of-service",
        "https://example.com/testimonials",
        "https://example.com/events",
        "https://example.com/gallery",
        "https://example.com/news",
        "https://example.com/subscribe",
        "https://example.com/resources",
        "https://example.com/support",
        "https://example.com/forum",
        "https://example.com/api-docs",
        "https://example.com/dashboard"
    ]
    
    Example output:
    {{
        "team_pages": ["https://example.com/team", "https://example.com/about", "https://example.com/contact"]
    }}

    Definition of a team page:

    A team page is a page which likely contains information about the team and hopefully contaact information such as an email address.
    """

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing URLs to find the best team pages. You return only JSON format",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    ai_response = json.loads(completion.choices[0].message.content)
    return ai_response.get("team_pages", [])


def find_team_pages(base_url: str) -> List[str]:
    """
    Find potential team pages on a website.
    First tries to use the sitemap, then falls back to crawling if necessary.
    """
    logger.info(f"Finding team pages for {base_url}")
    urls = fetch_sitemap(base_url)
    if not urls:
        logger.info("Sitemap not found or empty, falling back to crawling")
        urls = crawl_website(base_url, depth=2)
    team_pages = ai_select_team_pages(urls)
    logger.info(f"Found {len(team_pages)} potential team pages")
    return team_pages


def crawl_website(base_url: str, depth: int = 2) -> List[str]:
    """
    Crawl a website to a specified depth, collecting all unique URLs.
    """
    logger.info(f"Crawling website {base_url} to depth {depth}")
    visited = set()
    to_visit = [base_url]
    all_urls = set()

    for current_depth in range(depth):
        logger.info(f"Crawling depth {current_depth + 1}")
        next_to_visit = []
        for url in to_visit:
            if url in visited:
                continue
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                visited.add(url)
                soup = BeautifulSoup(response.text, "html.parser")
                links = soup.find_all("a", href=True)
                for link in links:
                    full_url = urljoin(base_url, link["href"])
                    if full_url.startswith(base_url):  # Only include URLs from the same domain
                        all_urls.add(full_url)
                        next_to_visit.append(full_url)
            except requests.RequestException as e:
                logger.warning(f"Error crawling {url}: {e}")
        to_visit = next_to_visit

    logger.info(f"Crawling complete. Found {len(all_urls)} unique URLs")
    return list(all_urls)


def process_website(row: Dict[str, str]) -> Dict[str, str]:
    """
    Process a single website: find team pages and extract emails.
    """
    website: str = row["website"]
    base_url = get_base_url(website)
    logger.info(f"Processing website: {base_url}")

    # Find team pages
    team_pages = find_team_pages(base_url)

    # Extract emails using the team pages
    emails = extract_emails_from_website(base_url, team_pages)

    row["Emails"] = ", ".join(emails)
    for i, page in enumerate(team_pages):
        row[f"TeamPage_{i+1}"] = page

    logger.info(f"Finished processing {base_url}. Found {len(emails)} emails and {len(team_pages)} team pages")
    return row


def process_websites_multithreaded(input_file: str, output_file: str, max_workers: int = 20) -> None:
    """
    Process multiple websites concurrently using multithreading.
    """
    logger.info(f"Starting multithreaded processing with {max_workers} workers")
    with open(input_file, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames: List[str] = reader.fieldnames + ["Emails"]
        for i in range(10):  # Assuming a maximum of 10 team pages
            fieldnames.append(f"TeamPage_{i+1}")

        total_rows = sum(1 for row in csv.DictReader(open(input_file)))
        infile.seek(0)
        next(reader)

        result_queue = queue.Queue()
        csv_lock = threading.Lock()

        def writer_thread():
            with open(output_file, "w", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                while True:
                    result = result_queue.get()
                    if result is None:
                        break
                    with csv_lock:
                        writer.writerow(result)
                        outfile.flush()
                    result_queue.task_done()

        writer = threading.Thread(target=writer_thread)
        writer.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in reader:
                future = executor.submit(process_website, row)
                future.add_done_callback(lambda f: result_queue.put(f.result()))
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=total_rows, desc="Processing websites", unit="website"):
                pass

        result_queue.put(None)
        writer.join()

    logger.info(f"Processing complete. Enriched data saved to {output_file}")


def enrich_website_data(input_file: str, output_file: str, max_workers: int = 20) -> None:
    """
    Main function to enrich website data with emails and team pages.
    """
    logger.info(f"Enriching website data from {input_file}")
    process_websites_multithreaded(input_file, output_file, max_workers)
    logger.info(f"Processing complete. Enriched data saved to {output_file}")


if __name__ == "__main__":
    enrich_website_data(
        "GooglePlaces/data/Websites_For_Scraping_Input.csv",
        "GooglePlaces/website-scraping-output-09-03-2024.csv",
        max_workers=30,
    )
