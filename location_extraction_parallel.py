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
client = OpenAI(api_key=e('OPENAI_API_KEY'))

file_path = "input/Location Information.csv"
output_file = 'location_extraction.csv'
batch_size = 20

def process_batch(batch_data):
    client = OpenAI(api_key=e('OPENAI_API_KEY'))
    
    location_input = batch_data.apply(lambda row: (
        f"[{row.name + 1}] Full Address: {row['FullPostalAddress']}, "
        f"Country: {row['Country']}, "
        f"City: {row['City']}, "
        f"Location Name: {row['LocationName']}, "
        f"State: {row['State']}, "
        f"Zip: {row['Zip']}, "
        f"Street Information: {row['Street1']} {row['Street2']}"
    ), axis=1).tolist()

    prompt = f"""I have a list of {len(batch_data)} addresses and details on the address. This data is not perfect so some of the fields may be blank, that is to be expected.
        
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

    Please replace (Full Address) with the exact FullPostalAddress - provided by Full Address: field that is given in this prompt and don't invent a new one. 
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
    
    return useful_format_answer

def main():
    our_data = pd.read_csv(file_path)
    num_batches = len(our_data) // batch_size + (1 if len(our_data) % batch_size else 0)
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_batch, [our_data.iloc[i:i+batch_size] for i in range(0, len(our_data), batch_size)]),
            total=num_batches,
            desc="Processing batches"
        ))
    
    # Combine results
    combined_data = {}
    for result in results:
        combined_data.update(result)
    
    # Writing to CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Full Address", "Categorization"])
        for key, value in combined_data.items():
            writer.writerow([key, value])
    
    print(f"Script completed successfully. Results written to {output_file}")

if __name__ == "__main__":
    main()