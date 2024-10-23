"""
This module classifies locations based on postal address information using OpenAI's GPT model.
It processes a CSV file containing location data and outputs a new CSV file with classifications.
"""

import csv
import json
from typing import Dict
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL
from tqdm import tqdm
import concurrent.futures

client = OpenAI(api_key=OPENAI_API_KEY)

EXAMPLES = """
Example 1:
Input:
FullPostalAddress: 123 Broadway, New York, NY 10001, United States

Output:
{
    "classification": "United States"
}

Example 2:
Input:
FullPostalAddress: 456 Yonge Street, Toronto, ON M4Y 1X9, Canada

Output:
{
    "classification": "North America"
}

Example 3:
Input:
FullPostalAddress: 10 Downing Street, London, SW1A 2AA, United Kingdom

Output:
{
    "classification": "EMEA"
}

Example 4:
Input:
FullPostalAddress: 1-1 Chiyoda, Tokyo 100-8111, Japan

Output:
{
    "classification": "APAC"
}

Example 5:
Input:
FullPostalAddress: Remote

Output:
{
    "classification": "Virtual"
}
"""


def classify_location(full_address: str) -> str:
    """
    Classify a location based on its full postal address.

    Args:
        full_address (str): The full postal address of the location.

    Returns:
        str: The classification of the location.

    Classification categories:
        1. United States
        2. North America (excluding the United States)
        3. Central America
        4. South America
        5. EMEA (Europe, Middle East, and Africa)
        6. APAC (Asia-Pacific)
        7. Virtual
    """
    prompt = f"""
    Given the following examples of location information and their classifications:

    {EXAMPLES}

    Now, analyze the following address:

    Input:
    FullPostalAddress: {full_address}

    Please respond with the classification in the following format:
    {{
        "classification": "(the most appropriate classification for this location)"
    }}
    
    Guidelines:
    - Use the following classification categories:
      (1) United States
      (2) North America (excluding the United States)
      (3) Central America
      (4) South America
      (5) EMEA (Europe, Middle East, and Africa)
      (6) APAC (Asia-Pacific)
      (7) Virtual
    - If the location is in the United States, classify it as "United States"
    - For Canada and Mexico, use "North America"
    - Use "EMEA" for locations in Europe, the Middle East, and Africa
    - Use "APAC" for locations in Asia and Oceania
    - If the address indicates a remote or virtual location, classify it as "Virtual"
    - If the location information is unclear or insufficient, use your best judgment based on available data

    VERY IMPORTANT:
    - DO NOT MAKE UP A CLASSIFICATION. YOU CAN ONLY SELECT FROM THE PROVIDED CATEGORIES.
    """

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing location information and determining the most appropriate classification. You return only JSON format.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    ai_response = json.loads(completion.choices[0].message.content)
    return ai_response.get("classification")


def process_location(row: Dict[str, str]) -> Dict[str, str]:
    """
    Process a single location row and classify it.

    Args:
        row (Dict[str, str]): A dictionary containing location information.

    Returns:
        Dict[str, str]: A dictionary with LocationID and Classification.
    """
    full_address = row.get("FullPostalAddress", "N/A")
    classification = classify_location(full_address)
    return {"LocationID": row.get("\ufeffLocationID", "Unknown"), "Classification": classification}


def process_csv(input_file_path: str, output_file_path: str, num_threads: int = 20):
    """
    Process the input CSV file, classify locations, and write results to the output CSV file.

    Args:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Path to the output CSV file.
        num_threads (int, optional): Number of threads to use for parallel processing. Defaults to 20.
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
        fieldnames = ["LocationID", "Classification"]
        writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Processing complete. Results written to {output_file_path}")


if __name__ == "__main__":
    input_csv_file_path = "GooglePlaces/data/location_information_10_16_2024.csv"
    output_csv_file_path = "GooglePlaces/data/location_information_10_16_2024_classified.csv"
    process_csv(input_file_path=input_csv_file_path, output_file_path=output_csv_file_path)
