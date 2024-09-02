import regex
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set, Tuple
import csv
import json
from pydantic import BaseModel, Field, HttpUrl
from urllib.parse import urlencode, urljoin, urlparse
from tqdm import tqdm
from config import GOOGLE_API_KEY, OPENAI_API_KEY, MODEL
from enum import Enum
from openai import OpenAI
import time
import concurrent.futures
import threading
import queue

client = OpenAI(api_key=OPENAI_API_KEY)


class APIChoice(Enum):
    KNOWLEDGE_GRAPH = "knowledge_graph"
    PLACES = "places"


class HotelInfo(BaseModel):
    name: str
    website: Optional[HttpUrl] = None
    url: Optional[HttpUrl] = None  # Google Maps URL
    formatted_address: Optional[str] = None
    adr_address: Optional[str] = None
    formatted_phone_number: Optional[str] = None
    international_phone_number: Optional[str] = None
    email: Optional[List[str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    plus_code: Optional[Dict[str, str]] = None
    types: Optional[List[str]] = None
    ai_email_inspection: Optional[str] = None
    room_number: Optional[str] = None
    explanation: Optional[str] = None


class KnowledgeGraphResponse(BaseModel):
    itemListElement: List[Dict] = Field(default_factory=list)


def get_base_url(url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


def extract_emails_from_text(text: str, chunk_size: int = 100000) -> List[str]:
    # Email pattern that matches most valid email formats
    email_pattern = regex.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    # Compile the cleaning patterns
    clean_start = regex.compile(r"^[^A-Za-z0-9]+")
    clean_end = regex.compile(r"[^A-Za-z0-9._%+-]+$")

    # Set of excluded extensions
    excluded_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

    valid_emails: Set[str] = set()

    # Process text in chunks
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]

        # Extract all matches based on the pattern
        matches = email_pattern.findall(chunk)

        for match in matches:
            # Clean any leading/trailing non-email characters
            cleaned = clean_start.sub("", match)
            cleaned = clean_end.sub("", cleaned)

            # Validate cleaned email against the email pattern
            if email_pattern.match(cleaned):
                # Extra check to exclude common non-email patterns
                if not any(cleaned.lower().endswith(ext) for ext in excluded_extensions):
                    valid_emails.add(cleaned)

    return list(valid_emails)


def ai_extract_email(text: str, max_length: int = 10000) -> Optional[str]:
    # Truncate the text to the maximum length specified
    if len(text) > max_length:
        text = text[:max_length] + "..."  # Add ellipsis to indicate truncation

    prompt = f"""
    Please respond in this format:
    {{
        "email": "(the email address you found)",
        "room_number": "(the number of rooms you found, answer N/A if not clear)",
        "how_sustainable": "(An explanation of how the hotel is sustainable, answer N/A if not clear)"
    }}
    
    If you find multiple email addresses, select the best email for contacting the hotel! 
    
    NOTE: DO NOT MAKE UP ANY INFORMATION - PLEASE RETURN A REAL EMAIL
    
    Extract the email address from the website text provided here:
    {text}
    """

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a text analyzing expert at finding the best emails for contacting on websites. You return only JSON format",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    ai_response = json.loads(completion.choices[0].message.content)
    return ai_response.get("email"), ai_response.get("room_number"), ai_response.get("how_sustainable")


def extract_emails_from_website(base_url: str, max_pages: int = 10) -> Tuple[List[str], str, str, str]:
    try:
        print(f"Extracting email from website: {base_url}")

        visited = set()
        to_visit = [base_url]
        all_emails = set()
        page_count = 0
        ai_email = None
        room_number = None
        explanation = None
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }

        website_base = get_base_url(base_url)
        base_domain = urlparse(base_url).netloc

        while to_visit and page_count < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue

            try:
                page_count += 1
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes

                content = ""
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    content += chunk.decode("utf-8", errors="ignore")
                    if len(content) > 10 * 1024 * 1024:  # Stop after 10MB
                        print(f"Page too large, stopping at 10MB: {url}")
                        break

                soup = BeautifulSoup(content, "html.parser")

                if url == base_url:
                    ai_email, room_number, explanation = ai_extract_email(content)

                page_emails = extract_emails_from_text(content)
                all_emails.update(page_emails)

                for link in tqdm(soup.find_all("a", href=True), desc=f"Processing links for {url}", leave=False):
                    href = link["href"]
                    if not href.startswith(("http://", "https://")):
                        full_url = urljoin(website_base, href)
                    else:
                        full_url = href

                    if urlparse(full_url).netloc == base_domain:
                        if full_url not in visited and full_url not in to_visit:
                            if any(term in href.lower() for term in ["about", "contact", "über", "kontakt", "uber"]):
                                to_visit.insert(0, full_url)  # Prioritize these pages
                            elif len(to_visit) < max_pages * 2:  # Limit size of to_visit
                                to_visit.append(full_url)

                visited.add(url)
                time.sleep(0.5)  # Increased delay for rate limiting
            except requests.RequestException as e:
                print(f"Error processing {url}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error processing {url}: {str(e)}")

        print(f"Found {len(all_emails)} email(s) from {page_count} pages")
        return list(all_emails), ai_email, room_number, explanation
    except Exception as e:
        print(f"Error extracting email from website: {e}")
        return [], None, None, None


def query_knowledge_graph(query: str) -> KnowledgeGraphResponse:
    service_url: str = "https://kgsearch.googleapis.com/v1/entities:search"
    params: Dict[str, str] = {
        "query": query,
        "limit": "5",
        "indent": "True",
        "key": GOOGLE_API_KEY,
        "prefix": True,
    }
    url: str = service_url + "?" + urlencode(params)
    response: requests.Response = requests.get(url)
    return KnowledgeGraphResponse(**json.loads(response.text))


def extract_hotel_info_kg(result: KnowledgeGraphResponse) -> Optional[HotelInfo]:
    if result.itemListElement:
        item: Dict = result.itemListElement[0].get("result", {})
        return HotelInfo(
            name=item.get("name", ""),
            description=item.get("description"),
            detailed_description=item.get("detailedDescription", {}).get("articleBody"),
            url=item.get("url"),
            image_url=item.get("image", {}).get("contentUrl"),
        )
    return None


def get_place_id(query: str) -> Optional[str]:
    base_url: str = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params: Dict[str, str] = {"input": query, "inputtype": "textquery", "key": GOOGLE_API_KEY}
    response: requests.Response = requests.get(base_url, params=params)
    result = response.json()
    if result.get("status") == "OK" and result.get("candidates"):
        return result["candidates"][0]["place_id"]
    return None


def get_place_details(place_id: str) -> Dict:
    base_url: str = "https://maps.googleapis.com/maps/api/place/details/json"
    params: Dict[str, str] = {
        "place_id": place_id,
        "fields": "name,website,formatted_address,formatted_phone_number,international_phone_number,geometry",
        "key": GOOGLE_API_KEY,
    }
    response: requests.Response = requests.get(base_url, params=params)
    return response.json()


def extract_hotel_info_places(place_details: Dict) -> Optional[HotelInfo]:
    if "result" in place_details:
        result = place_details["result"]

        email = None
        ai_email_inspection = None
        room_number = None
        explanation = None

        # Try to find email in adr_address
        if "adr_address" in result:
            email = [extract_emails_from_text(result["adr_address"])]

        # If no email found and website is available, try to extract from website
        if not email and "website" in result:
            email, ai_email_inspection, room_number, explanation = extract_emails_from_website(result["website"])

        return HotelInfo(
            name=result.get("name", ""),
            website=result.get("website"),
            url=result.get("url"),
            formatted_address=result.get("formatted_address"),
            adr_address=result.get("adr_address"),
            formatted_phone_number=result.get("formatted_phone_number"),
            international_phone_number=result.get("international_phone_number"),
            email=email,
            latitude=result.get("geometry", {}).get("location", {}).get("lat"),
            longitude=result.get("geometry", {}).get("location", {}).get("lng"),
            plus_code=result.get("plus_code"),
            types=result.get("types"),
            ai_email_inspection=ai_email_inspection,
            room_number=room_number,
            explanation=explanation,
        )
    return None


def process_hotel(row: Dict[str, str], api_choice: APIChoice) -> Dict[str, str]:
    hotel_name: str = row["HotelName"]
    address: str = row.get("address", "")

    # Construct query: "{hotelName} {address}" or "{hotelName} Germany" if no address
    query = f"{hotel_name} {address}" if address else f"{hotel_name} Germany"

    print(f"processing: {hotel_name}")
    print(f"using query: {query}")

    hotel_info = None

    if api_choice == APIChoice.KNOWLEDGE_GRAPH:
        result: KnowledgeGraphResponse = query_knowledge_graph(query)
        hotel_info: Optional[HotelInfo] = extract_hotel_info_kg(result)
    elif api_choice == APIChoice.PLACES:
        place_id = get_place_id(query)
        if place_id:
            place_details = get_place_details(place_id)
            hotel_info = extract_hotel_info_places(place_details)
    else:
        raise ValueError("Invalid API choice. Choose from KNOWLEDGE_GRAPH and PLACES")

    if hotel_info is not None:
        row.update(hotel_info.dict(exclude_unset=True))

    return row


def process_hotels_multithreaded(input_file: str, output_file: str, api_choice: APIChoice, max_workers: int = 20) -> None:
    with open(input_file, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames: List[str] = reader.fieldnames + [
            "name",
            "website",
            "url",
            "formatted_address",
            "adr_address",
            "formatted_phone_number",
            "international_phone_number",
            "email",
            "latitude",
            "longitude",
            "plus_code",
            "types",
            "ai_email_inspection",
            "room_number",
            "explanation",
        ]

        total_rows = sum(1 for row in csv.DictReader(open(input_file)))
        infile.seek(0)
        next(reader)  # Skip the header row

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
                        outfile.flush()  # Ensure data is written to disk
                    result_queue.task_done()

        writer = threading.Thread(target=writer_thread)
        writer.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in reader:
                future = executor.submit(process_hotel, row, api_choice)
                future.add_done_callback(lambda f: result_queue.put(f.result()))
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=total_rows, desc="Processing hotels", unit="hotel"):
                pass  # Progress bar

        # Signal the writer thread to finish
        result_queue.put(None)
        writer.join()

    print(f"Processing complete. Enriched data saved to {output_file}")


def enrich_hotel_data(input_file: str, output_file: str, api_choice: APIChoice = APIChoice.PLACES, max_workers: int = 20) -> None:
    """
    Main function to enrich hotel data from an input CSV file and save to an output CSV file.

    :param input_file: Path to the input CSV file
    :param output_file: Path to the output CSV file where enriched data will be saved
    :param api_choice: Choose between KNOWLEDGE_GRAPH, PLACES, and OSM API
    """
    process_hotels_multithreaded(input_file, output_file, api_choice, max_workers)
    print(f"Processing complete using {api_choice.value}. Enriched data saved to {output_file}")


# This block will only run if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    # Example usage
    enrich_hotel_data(
        "lead_generation_modules/googlePlaces/remaining_hotels_with_some_duplicates.csv",
        "lead_generation_modules/googlePlaces/remaining_hotels_with_some_duplicates_output.csv",
        # "lead_generation_sessions/hotels-08-08-24/TestHotelsBatch.csv",
        # "lead_generation_sessions/hotels-08-08-24/testFinalv4.csv",
        APIChoice.PLACES,
        max_workers=30,
    )
