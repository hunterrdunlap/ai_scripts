import requests
import pandas as pd
import os
import time
from tqdm import tqdm

from config import GOOGLE_API_KEY

# Base URL for the Google Places API (using Text Search)
PLACES_API_BASE_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"


# --- Function to get Place Details from the Places API ---
def get_place_details(address):
    """
    Fetches place details from the Google Places API using the Text Search endpoint.

    Args:
        address (str): The full postal address to search for.

    Returns:
        dict: A dictionary containing place details, or None if an error occurs.
    """

    params = {
        "query": address,
        "key": GOOGLE_API_KEY,
    }

    try:
        response = requests.get(PLACES_API_BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()

        if data["status"] == "OK" and data["results"]:
            # Extract relevant details (customize as needed)
            place = data["results"][0]  # Get the first result (usually the most relevant)
            place_id = place.get("place_id")
            formatted_address = place.get("formatted_address")
            location_type = place.get("types", [])  # Get the place types
            rating = place.get("rating")
            user_ratings_total = place.get("user_ratings_total")

            # Get latitude and longitude
            location = place.get("geometry", {}).get("location", {})
            latitude = location.get("lat")
            longitude = location.get("lng")

            # You can add more fields here if needed.

            return {
                "place_id": place_id,
                "formatted_address": formatted_address,
                "location_type": location_type,
                "rating": rating,
                "user_ratings_total": user_ratings_total,
                "latitude": latitude,
                "longitude": longitude,
            }

        elif data["status"] == "ZERO_RESULTS":
            print(f"No results found for address: {address}")
            return None

        else:
            print(f"Error for address: {address}. Status: {data['status']}")
            if "error_message" in data:
                print(f"Error message: {data['error_message']}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request error for address: {address}: {e}")
        return None

    except ValueError as e:
        print(f"JSON decoding error for address: {address}: {e}")
        return None


# --- Main script ---
def main():
    """
    Reads a CSV file with location IDs and addresses, fetches place details from the Google Places API,
    and saves the appended data to a new CSV file.
    """

    input_csv_file = "/Users/hunterdunlap/Downloads/advoda-location-info.csv"  # Replace with your input CSV file name
    output_csv_file = "advoda-location-info-enhanced.csv"  # Replace with your desired output CSV file name

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv_file)

        # let's start with just the first 10 rows
        df = df.head(10)

        # Check if the required columns exist
        if not {"LocationID", "FullPostalAddress"}.issubset(df.columns):
            raise ValueError("CSV file must contain 'LocationID' and 'FullPostalAddress' columns.")

        # Create empty lists to store the appended data
        place_ids = []
        formatted_addresses = []
        location_types = []
        ratings = []
        user_ratings_totals = []
        latitudes = []
        longitudes = []

        # Iterate through each row in the DataFrame with tqdm progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing addresses"):
            location_id = row["LocationID"]
            address = row["FullPostalAddress"]

            print(f"Processing locationID: {location_id}, Address: {address}")

            place_details = get_place_details(address)

            if place_details:
                place_ids.append(place_details["place_id"])
                formatted_addresses.append(place_details["formatted_address"])
                location_types.append(place_details["location_type"])
                ratings.append(place_details["rating"])
                user_ratings_totals.append(place_details["user_ratings_total"])
                latitudes.append(place_details["latitude"])
                longitudes.append(place_details["longitude"])

            else:
                # Append None or empty values if no details are found
                place_ids.append(None)
                formatted_addresses.append(None)
                location_types.append(None)
                ratings.append(None)
                user_ratings_totals.append(None)
                latitudes.append(None)
                longitudes.append(None)

            time.sleep(0.2)  # Add a small delay to avoid hitting rate limits

        # Add the new columns to the DataFrame
        df["place_id"] = place_ids
        df["formatted_address"] = formatted_addresses
        df["location_type"] = location_types
        df["rating"] = ratings
        df["user_ratings_total"] = user_ratings_totals
        df["latitude"] = latitudes
        df["longitude"] = longitudes

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_csv_file, index=False)

        print(f"Data successfully saved to {output_csv_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_file}' not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
