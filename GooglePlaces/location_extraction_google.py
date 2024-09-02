import csv
import requests
from typing import Dict, Optional, List
from pydantic import BaseModel
from tqdm import tqdm
import concurrent.futures
from config import GOOGLE_API_KEY

MAX_WORKERS = 10


class AddressInfo(BaseModel):
    """Pydantic model for storing address information."""

    formatted_address: Optional[str] = None


def get_place_id(query: str) -> Optional[str]:
    """
    Retrieve the place ID for a given address query using Google Places API.

    Args:
        query (str): The address to search for.

    Returns:
        Optional[str]: The place ID if found, None otherwise.
    """
    base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {"input": query, "inputtype": "textquery", "key": GOOGLE_API_KEY}
    response = requests.get(base_url, params=params)
    result = response.json()
    if result.get("status") == "OK" and result.get("candidates"):
        return result["candidates"][0]["place_id"]
    return None


def get_place_details(place_id: str) -> Dict:
    """
    Retrieve place details for a given place ID using Google Places API.

    Args:
        place_id (str): The place ID to look up.

    Returns:
        Dict: A dictionary containing place details.
    """
    base_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "fields": "formatted_address", "key": GOOGLE_API_KEY}
    response = requests.get(base_url, params=params)
    return response.json()


def process_address(row: Dict[str, str]) -> Dict[str, str]:
    """
    Process a single address row, enriching it with Google's formatted address.

    Args:
        row (Dict[str, str]): A dictionary representing a row from the input CSV.

    Returns:
        Dict[str, str]: The input row enriched with Google's formatted address.
    """
    query = f"{row['FullAddress']}"
    place_id = get_place_id(query)
    if place_id:
        place_details = get_place_details(place_id)
        if "result" in place_details:
            formatted_address = place_details["result"].get("formatted_address")
            row["GoogleFormattedAddress"] = formatted_address
    return row


def process_batch(batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Process a batch of address rows.

    Args:
        batch (List[Dict[str, str]]): A list of rows to process.

    Returns:
        List[Dict[str, str]]: A list of processed rows.
    """
    return [process_address(row) for row in batch]


def enrich_address_data(input_file: str, output_file: str) -> None:
    """
    Enrich address data from an input CSV file with Google's formatted addresses
    and write the results to an output CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
    """
    with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["GoogleFormattedAddress"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Read all rows and prepare batches for parallel processing
        all_rows = list(reader)
        total_rows = len(all_rows)
        batch_size = 10
        batches = [all_rows[i : i + batch_size] for i in range(0, total_rows, batch_size)]

        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
                for enriched_row in future.result():
                    writer.writerow(enriched_row)

    print(f"Processing complete. Enriched data saved to {output_file}")


if __name__ == "__main__":
    enrich_address_data("GooglePlaces/services_location_data.csv", "GooglePlaces/services_location_data_with_google_format.csv")
