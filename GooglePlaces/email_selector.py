import csv
import json
from typing import Dict
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL
from tqdm import tqdm
import concurrent.futures
import ast

client = OpenAI(api_key=OPENAI_API_KEY)

EXAMPLES = """
Example 1:
Input:
Hotel Name: Ringhotel Schorfheide Tagungszentrum der Wirtschaft Hubertusstock 2 16247 Joachimsthal Germany
Email(s): alina.loesche@tagungs-zentrum.de, verkauf@tagungs-zentrum.de, verwaltung@tagungs-zentrum.de, martin.braun@tagungs-zentrum.de, rezeption@tagungs-zentrum.de
Website: http://www.tagungs-zentrum.de/
Address: Hubertusstock 2, 16247 Joachimsthal, Germany

Output:
{
    "best_email": "rezeption@tagungs-zentrum.de"
}

Example 2:
Input:
Hotel Name: Amedia Plaza Dresden Juedenhof 9 Altstadt 01067 Dresden Germany
Email(s): accessibility@wyndham.com, dresdenplaza@amediahotels.com, johnsmith@email.com
Website: https://www.wyndhamhotels.com/trademark/dresden-germany/amedia-plaza-dresden-trademark-collection/overview?CID=LC:4d7auublw4ufaga:57354&iata=00093796
Address: Jüdenhof 9, 01067 Dresden, Germany

Output:
{
    "best_email": "dresdenplaza@amediahotels.com"
}

Example 3:
Input:
Hotel Name: Vienna House By Wyndham Remarque Osnabru Natruper-Tor-Wall 1 49076 Osnabrück Germany
Email(s): accessibility@wyndham.com, reception.remarque-osnabrueck@hrg-hotels.com, privacy@wyndham.com
Website: https://www.wyndhamhotels.com/de-de/vienna-house/osnabrueck-germany/vienna-house-remarque-osnabruck/overview?CID:LC:6qt7c54dekbf1g7:57387
Address: Natruper-Tor-Wall 1, 49076 Osnabrück, Germany

Output:
{
    "best_email": "reception.remarque-osnabrueck@hrg-hotels.com"
}

Example 4:
Input:
Hotel Name: Munich Marriott Hotel Berliner Str. 93 Schwabing-Freimann 80805 Munich Germany
Email(s): associateprivacy@marriott.com, privacy@marriott.com, applicantprivacyinfo@marriott.com, muenchen.marriott@marriotthotels.com, gen-mucnomrktcommgr-DL@marriott.com, MarriottDPO@marriott.com, Heiraten-in-muenchen@marriott.com, info@championsbar.de, Munich.salesoffice@marriott.com
Website: https://www.marriott.com/en-us/hotels/mucno-munich-marriott-hotel/overview/?scid=f2ae0541-1279-4f24-b197-a979c79310b0
Address: Berliner Str. 93, 80805 München, Germany

Output:
{
    "best_email": "muenchen.marriott@marriotthotels.com"
}

Example 5:
Input:
Hotel Name: Waldhotel am Notschreipass Notschrei Passhöhe 2 79674 Todtnau Germany
Email(s): ba@albiez-team.de, info@schwarzwald-waldhotel.de
Website: https://www.schwarzwald-waldhotel.de/de/
Address: Notschrei-Passhöhe 2, 79674 Todtnau, Germany

Output:
{
    "best_email": "info@schwarzwald-waldhotel.de"
}
"""


def select_best_email(hotel_name: str, email_list: list, hotel_info: dict) -> str:
    email_str = ", ".join(email_list) if email_list else "N/A"

    prompt = f"""
    Given the following examples of hotel information and their selected best email addresses:

    {EXAMPLES}

    Now, analyze the following information for a new hotel:

    Input:
    Hotel Name: {hotel_name}
    Email(s): {email_str}
    Website: {hotel_info.get('website', 'N/A')}
    Address: {hotel_info.get('formatted_address', 'N/A')}

    Please respond with the best email in the following format:
    {{
        "best_email": "(the most appropriate email address for contacting the hotel)"
    }}
    
    Guidelines:
    - If multiple email addresses are provided, prioritize in the following order:
    1. Emails containing 'manager', 'director', or 'gm' (for General Manager)
    2. Emails containing the hotel's name, 'reception', or 'info'
    3. Emails containing 'sales', 'marketing', or 'operations'
    - For chain hotels, prefer an email specific to this location rather than a general corporate email.
    - Avoid selecting emails related to privacy, accessibility, or specific departments (e.g., housekeeping, restaurant) unless they're the only option.
    - For smaller or independent hotels, 'owner' or 'reservations' emails might also be appropriate.
    - If the hotel name includes words like "resort" or "spa", consider prioritizing emails with 'management' or 'director'.
    - Avoid emails that seem to be for individual employees (e.g., john.smith@hotel.com) unless no other options are available.
    - If no suitable email can be determined, return "N/A".
    - Use the examples provided to guide your decision, considering patterns in how email addresses are formed for different types of hotels.

    VERY IMPORTANT:
    - DO NOT MAKE UP AN EMAIL. YOU CAN ONLY SELECT FROM THE PROVIDED LIST. MAKING UP AN EMAIL WILL CREATE INCORRECT DATA.
    """

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing hotel information and determining the most appropriate contact email. You return only JSON format.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    ai_response = json.loads(completion.choices[0].message.content)
    return ai_response.get("best_email")


def safe_eval_list(value):
    if not value or value.strip() == "":
        return []
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else [str(result)]
    except (ValueError, SyntaxError):
        return [value.strip()]


def process_hotel(row: Dict[str, str]) -> Dict[str, str]:
    hotel_name = row.get("HotelName", "Unknown Hotel")
    email_list = safe_eval_list(row.get("email", ""))
    hotel_info = {
        "website": row.get("website", "N/A"),
        "formatted_address": row.get("formatted_address", "N/A"),
    }
    best_email = select_best_email(hotel_name, email_list, hotel_info)
    return {"HotelName": hotel_name, "SelectedEmail": best_email}


def process_csv(input_file_path: str, output_file_path: str, num_threads: int = 20):
    # Read all rows from the input CSV file
    with open(input_file_path, newline="", encoding="utf-8") as input_csvfile:
        reader = csv.DictReader(input_csvfile)
        rows = list(reader)

    # Process hotels in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_hotel, rows), total=len(rows), desc="Processing hotels", unit="hotel"))

    # Write results to the output CSV file
    with open(output_file_path, "w", newline="", encoding="utf-8") as output_csvfile:
        fieldnames = ["HotelName", "SelectedEmail"]
        writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Processing complete. Results written to {output_file_path}")


if __name__ == "__main__":
    input_csv_file_path = "lead_generation_modules/googlePlaces/remaining_hotels_with_some_duplicates_output.csv"
    output_csv_file_path = "lead_generation_modules/googlePlaces/remaining_hotels_with_some_duplicates_output_emails_selected.csv"
    process_csv(input_csv_file_path, output_csv_file_path)
