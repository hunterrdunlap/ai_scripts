import ast
import csv
from os import environ as env

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
e = env.get
client = OpenAI(api_key=e('OPENAI_API_KEY'))

file_path = "input/Location Information.csv"

our_data = pd.read_csv(file_path)

data = {}

# Iterate over 20 rows at a time
for i in tqdm(range(0, len(our_data), 20)):
    # Get the next 20 rows
    next_batch = our_data.iloc[i:i+20]

    # Create location input string
    location_input = next_batch.apply(lambda row: (
        f"Full Address: {row['FullPostalAddress']}, "
        f"Country: {row['Country']}, "
        f"City: {row['City']}, "
        f"Location Name: {row['LocationName']}, "
        f"State: {row['State']}, "
        f"Zip: {row['Zip']}, "
        f"Street Information: {row['Street1']} {row['Street2']}"
    ), axis=1).tolist()

    prompt = f"""I have a list of 20 addresses and details on the address. This data is not perfect so some of the fields may be blank, that is to be expected.
        
        You have one objective with this information, determine the categorization based on these 6 options:

        (1) United States
        (2) North America 
            - note: this excludes the United States
        (3) Central America
        (4) South America
        (5) EMEA
        (6) APAC
        (7) Virtual

    
    Please be careful and diligent while reviewing these addresses. Do not skip any addresses. The ordering is extremely important. 
                ```
                {location_input},
                ```

                Please format your response as a traditional JSON format as follows:
                ```
                {{ "(Full Address)": [(Categorization)],
                   "(Full Address)": [(Categorization)],
                   "(Full Address)": [(Categorization)],
                   ...
                   }},
                ```
                
                
    For example, the output might look like this:
    ```
    {{ "Some Address 1": ["United States"],
         "Some Address 2": ["North America"],
         "Some Address 3": ["Central America"],
         "Some Address 4": ["South America"],
         "WebEx  (AWS) Virtual Office (VO), ., . .": ["Virtual"],
         ...
         }},
     ```
    

    Please replace (Full Address) with the exact full address that is given in this prompt and don't invent a new one. 
    IT IS VERY IMPORTANT THAT THE FULL ADDRESS STAYS EXACTLY THE SAME. DON'T CHANGE IT AT ALL
    Please also note that the commas at the end of each response are important. 
    Please answer for every location - there is no reason to not do all of them. 
    VERY IMPORTANT: Only answer in the JSON format say no more or less than the JSON format! This is being used in a script and any additional text will break it. 
    DO NOT SAY ANYTHING ELSE OTHER THAN THE JSON FORMAT AND ANSWER FOR ALL COMPANIES.
    IMPORTANT: DO NOT WRAP ANSWER IN JSON``` ```, JUST ANSWER AND START WITH CURLY BRACE - NOTHING ELSE. 
    """

    def get_answer(prompt):
        completion = client.chat.completions.create(
            model=e('MODEL'),
            messages=[
                {"role": "system", "content": "You are a address analyzer and answerer who only responds with the appropriate JSON format. You give the best guess based on the information provided and follow the JSON format exactly."},
                {"role": "user", "content": f"{prompt}"},
            ]
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
    try:
        data.update(useful_format_answer)
    except ValueError:
        try:
            # Possibly returned this single list thing...
            data.update(useful_format_answer[0])
        except:
            raise ValueError("Unexpected response format")

# Writing to CSV
filename = 'location_extraction.csv'

def write_dict_to_csv(input_dict, file_name):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Full Address", "Categorization"])
        for key, value in input_dict.items():
            categorization = value
            writer.writerow([key, categorization])

write_dict_to_csv(data, filename)
print("Script completed successfully")