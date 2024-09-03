import regex
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set, Tuple
import csv
import json
from pydantic import BaseModel, Field, HttpUrl
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from config import GOOGLE_API_KEY, OPENAI_API_KEY, MODEL
from enum import Enum
from openai import OpenAI
import time
import concurrent.futures
import threading
import queue

client = OpenAI(api_key=OPENAI_API_KEY)


class HotelInfo(BaseModel):
    name: str
    website: Optional[HttpUrl] = None
    formatted_address: Optional[str] = None
    international_phone_number: Optional[str] = None
    national_phone_number: Optional[str] = None  # Added field
    email: Optional[List[str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    ai_email_inspection: Optional[str] = None
    room_number: Optional[str] = None
    explanation: Optional[str] = None


def get_base_url(url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


def extract_emails_from_text(text: str, chunk_size: int = 100000) -> List[str]:
    email_pattern = regex.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    clean_start = regex.compile(r"^[^A-Za-z0-9]+")
    clean_end = regex.compile(r"[^A-Za-z0-9._%+-]+$")
    excluded_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    valid_emails: Set[str] = set()

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        matches = email_pattern.findall(chunk)
        for match in matches:
            cleaned = clean_start.sub("", match)
            cleaned = clean_end.sub("", cleaned)
            if email_pattern.match(cleaned):
                if not any(cleaned.lower().endswith(ext) for ext in excluded_extensions):
                    valid_emails.add(cleaned)

    return list(valid_emails)


def ai_extract_email(text: str, max_length: int = 10000) -> Optional[str]:
    if len(text) > max_length:
        text = text[:max_length] + "..."

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
                response.raise_for_status()

                content = ""
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    content += chunk.decode("utf-8", errors="ignore")
                    if len(content) > 10 * 1024 * 1024:
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
                                to_visit.insert(0, full_url)
                            elif len(to_visit) < max_pages * 2:
                                to_visit.append(full_url)

                visited.add(url)
                time.sleep(0.5)
            except requests.RequestException as e:
                print(f"Error processing {url}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error processing {url}: {str(e)}")

        print(f"Found {len(all_emails)} email(s) from {page_count} pages")
        return list(all_emails), ai_email, room_number, explanation
    except Exception as e:
        print(f"Error extracting email from website: {e}")
        return [], None, None, None


def query_places_api(query: str) -> Optional[HotelInfo]:
    base_url: str = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.websiteUri,places.internationalPhoneNumber,places.nationalPhoneNumber,places.location",  # Added field
    }
    data = json.dumps({"textQuery": query})
    response: requests.Response = requests.post(base_url, headers=headers, data=data)
    result = response.json()

    if result.get("places"):
        place = result["places"][0]
        email = None
        ai_email_inspection = None
        room_number = None
        explanation = None

        # if "websiteUri" in place:
        #     email, ai_email_inspection, room_number, explanation = extract_emails_from_website(place["websiteUri"])

        return HotelInfo(
            name=place.get("displayName", {}).get("text", ""),
            website=place.get("websiteUri"),
            formatted_address=place.get("formattedAddress"),
            international_phone_number=place.get("internationalPhoneNumber"),
            national_phone_number=place.get("nationalPhoneNumber"),  # Added field
            email=email,
            latitude=place.get("location", {}).get("latitude"),
            longitude=place.get("location", {}).get("longitude"),
            ai_email_inspection=ai_email_inspection,
            room_number=room_number,
            explanation=explanation,
        )
    return None


def process_hotel(row: Dict[str, str]) -> Dict[str, str]:
    hotel_name: str = row["HotelName"]
    query = f"{hotel_name} deutschland"

    print(f"processing: {hotel_name}")
    print(f"using query: {query}")

    hotel_info = None

    hotel_info = query_places_api(query)

    if hotel_info is not None:
        row.update(hotel_info.dict(exclude_unset=True))

    return row


def process_hotels_multithreaded(input_file: str, output_file: str, max_workers: int = 20) -> None:
    with open(input_file, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames: List[str] = reader.fieldnames + [
            "name",
            "website",
            "formatted_address",
            "international_phone_number",
            "national_phone_number",  # Added field
            "email",
            "latitude",
            "longitude",
            "ai_email_inspection",
            "room_number",
            "explanation",
        ]

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
                future = executor.submit(process_hotel, row)
                future.add_done_callback(lambda f: result_queue.put(f.result()))
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=total_rows, desc="Processing hotels", unit="hotel"):
                pass

        result_queue.put(None)
        writer.join()

    print(f"Processing complete. Enriched data saved to {output_file}")


def enrich_hotel_data(input_file: str, output_file: str, max_workers: int = 20) -> None:
    process_hotels_multithreaded(input_file, output_file, max_workers)
    print(f"Processing complete. Enriched data saved to {output_file}")


if __name__ == "__main__":
    enrich_hotel_data(
        "GooglePlaces/Hotel_Names_That_Did_Not_Work.csv",
        "GooglePlaces/enriched_hotel_names_try_again-09-02-2024.csv",
        max_workers=20,
    )
