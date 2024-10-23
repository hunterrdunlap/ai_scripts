import csv
import os
from typing import Dict, List
import json
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL
from tqdm import tqdm
import concurrent.futures

client = OpenAI(api_key=OPENAI_API_KEY)

EXAMPLES = """
Example 1:
Input:
Address: 152 The Arches Circle Suite 912, Deer Park, NY 11729

Output:
{
    "category": "Retail Store"
}

Example 2:
Input:
Address: 4060 E Francis, Suite 200, Ontario, CA 91761

Output:
{
    "category": "Office"
}

Example 3:
Input:
Address: 10451 Dog Leg Road, Vandalia, OH 45414

Output:
{
    "category": "Distribution Center"
}

Example 4:
Input:
Address: Crocs Industrial(ShenZhen) CO., LTD Village, Longgang District, Shenzhen City

Output:
{
    "category": "Manufacturing Facility"
}

Example 5:
Input:
Address: WebEx (AWS) Virtual Office (VO), ., . .

Output:
{
    "category": "Other"
}
"""


def categorize_address(address: str) -> str:
    """
    Categorize an address using the OpenAI API.

    Args:
        address (str): The full postal address to categorize.

    Returns:
        str: The category assigned to the address.

    Categories:
        1. Retail Store
        2. Office
        3. Warehouse
        4. Manufacturing Facility
        5. Distribution Center
        6. Other
    """
    prompt = f"""
    Given the following examples of addresses and their categories:

    {EXAMPLES}

    Now, analyze the following address:

    Input:
    Address: {address}

    Please respond with the category in the following format:
    {{
        "category": "(the most appropriate category for this address)"
    }}
    
    Guidelines:
    - Use the following categories only:
      (1) Retail Store - For shopping centers, malls, outlets, and standalone stores
      (2) Office - For corporate offices, business centers, and professional spaces
      (3) Warehouse - For storage facilities and warehouses
      (4) Manufacturing Facility - For factories and production facilities
      (5) Distribution Center - For logistics and distribution facilities
      (6) Other - For virtual offices, PO boxes, or addresses that don't fit other categories
    
    VERY IMPORTANT:
    - DO NOT MAKE UP A CATEGORY. YOU CAN ONLY SELECT FROM THE PROVIDED CATEGORIES.
    - Look for keywords like "Suite", "Mall", "Industrial", "Factory", etc.
    - Consider the location context and address format
    """

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing addresses and determining their facility type. You return only JSON format.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    ai_response = json.loads(completion.choices[0].message.content)
    return ai_response.get("category")


def process_location(row: Dict[str, str]) -> Dict[str, str]:
    """
    Process a single location row and categorize it.

    Args:
        row (Dict[str, str]): A dictionary containing location information.

    Returns:
        Dict[str, str]: A dictionary with LocationID, FullPostalAddress, and Category.
    """
    address = row.get("FullPostalAddress", "N/A")
    category = categorize_address(address)
    return {"LocationID": row.get("LocationID", "Unknown"), "FullPostalAddress": address, "Category": category}


def process_csv(input_file_path: str, output_file_path: str, num_threads: int = 20):
    """
    Process the input CSV file, categorize locations, and write results to the output CSV file.

    Args:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Path to the output CSV file.
        num_threads (int, optional): Number of threads for parallel processing. Defaults to 20.
    """
    # Read all rows from the input CSV file
    with open(input_file_path, newline="", encoding="utf-8") as input_csvfile:
        reader = csv.DictReader(input_csvfile)
        rows = list(reader)

    # Process locations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(iterable=executor.map(process_location, rows), total=len(rows), desc="Processing locations", unit="location"))

    # Write results to the output CSV file
    with open(output_file_path, "w", newline="", encoding="utf-8") as output_csvfile:
        fieldnames = ["LocationID", "FullPostalAddress", "Category"]
        writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Processing complete. Results written to {output_file_path}")


if __name__ == "__main__":
    input_csv_path = "data/location_information_10_16_2024.csv"
    output_csv_path = "data/categorized_locations.csv"
    process_csv(input_file_path=input_csv_path, output_file_path=output_csv_path)
