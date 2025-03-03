import ast
import csv
from os import environ as env
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import multiprocessing as mp

load_dotenv()
e = env.get
client = OpenAI(api_key=e("OPENAI_API_KEY"))

# file_path = "/Users/hunterdunlap/Downloads/advoda-location-info.csv"
file_path = "/Users/hunterdunlap/Downloads/null-locations.csv"
output_file = "location_extraction_skeptical_null.csv"
batch_size = 20


def process_batch(batch_data):
    client = OpenAI(api_key=e("OPENAI_API_KEY"))

    location_input = batch_data.apply(
        lambda row: (
            f"LocationID: [{row['LocationID']}], "
            f"Full Address: {row['FullPostalAddress']}, "
            f"Country: {row['Country']}, "
            f"City: {row['City']}, "
            f"Location Name: {row['LocationName']}, "
            f"State: {row['State']}, "
            f"Zip: {row['Zip']}, "
            f"Street Information: {row['Street1']} {row['Street2']}"
        ),
        axis=1,
    ).tolist()

    prompt = f"""I have a list of {len(batch_data)} addresses and details on the address. This data is not perfect so some of the fields may be blank, that is to be expected.
        
        You have one objective with this information, determine the categorization based on these 6 options:

        (1) North America 
        (2) Central America
        (3) South America
        (4) EMEA
        (5) APAC

    
    Please be careful and diligent while reviewing these addresses. Do not skip any addresses. The ordering is extremely important. 
                ```
                {location_input},
                ```

                Please format your response as a traditional JSON format as follows, where the key is the location ID:
                ```
                {{ 
                    "1": {{
                        "address": "full_postal_address",
                        "category": "categorization"
                    }},
                    "2": {{
                        "address": "full_postal_address",
                        "category": "categorization"
                    }}
                }}
                ```
                
                
    For example, the output might look like this:
    ```
    {{ 
        "1": {{
            "address": "123 Main St, New York, NY",
            "category": "United States"
        }},
        "2": {{
            "address": "456 Queen St, Toronto, ON",
            "category": "North America"
        }},
        "3": {{
            "address": "789 Mexico City",
            "category": "Central America"
        }}
    }}
    ```

    Please use the exact location ID (number in square brackets) and FullPostalAddress from the input.
    Please answer for every location - there is no reason to not do all of them. 
    VERY IMPORTANT: Only answer in the JSON format say no more or less than the JSON format! This is being used in a script and any additional text will break it. 
    DO NOT SAY ANYTHING ELSE OTHER THAN THE JSON FORMAT AND ANSWER FOR ALL LOCATIONS.
    IMPORTANT: DO NOT WRAP ANSWER IN JSON``` ```, JUST ANSWER AND START WITH CURLY BRACE - NOTHING ELSE. 
    """

    def get_answer(prompt):
        completion = client.chat.completions.create(
            model=e("MODEL"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a address analyzer and answerer who only responds with the appropriate JSON format. You give the best guess based on the information provided and follow the JSON format exactly.",
                },
                {"role": "user", "content": f"{prompt}"},
            ],
        )
        answer = completion.choices[0].message
        useful_format_answer = None
        try:
            useful_format_answer = ast.literal_eval(answer.content)
        except:
            print("[ALERT] Data was not formatted in expected way")
            print("Bad Answer: \n \n \n", answer)
        return useful_format_answer

    useful_format_answer = None
    while useful_format_answer is None:
        useful_format_answer = get_answer(prompt)

    return useful_format_answer


def main():
    our_data = pd.read_csv(file_path)
    # our_data = our_data.head(100)
    num_batches = len(our_data) // batch_size + (1 if len(our_data) % batch_size else 0)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(process_batch, [our_data.iloc[i : i + batch_size] for i in range(0, len(our_data), batch_size)]),
                total=num_batches,
                desc="Processing batches",
            )
        )

    # Combine results
    combined_data = {}
    for result in results:
        combined_data.update(result)

    # Writing to CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Location ID", "Full Address", "Categorization"])
        for key, value in combined_data.items():
            writer.writerow([key, value["address"], value["category"]])

    print(f"Script completed successfully. Results written to {output_file}")


if __name__ == "__main__":
    main()
