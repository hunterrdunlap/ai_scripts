# main.py
from search_engine import get_search_results, get_page_content
from llm_interface import generate_search_query, generate_answer
from csv_handler import read_csv, write_json_output
from config import BRAVE_API_KEY
from typing import Dict, List, Any
from tqdm import tqdm

def process_csv_row(row: Dict[str, str], prompt: str, query_prompt: str, input_file_information: str, response_format: str) -> Dict[str, Any]:
    # Generate search query
    query = generate_search_query(row, prompt, query_prompt)
    
    # Get search results
    search_results = get_search_results(query, BRAVE_API_KEY)
    
    # Fetch page contents
    page_contents = []
    for result in search_results:
        content = get_page_content(result['url'])
        page_contents.append({
            'url': result['url'],
            'content': content
        })
    
    # Generate answer using Claude
    answer = generate_answer(row, page_contents, prompt, input_file_information, response_format)
    
    return answer

def main(input_file: str, output_file: str) -> None:
    rows = read_csv(input_file)
    results = []
    
    prompt = """I need to know three things about this company:
            (1) How many full time employees does the company have? 
            (2) Where is the company located? City, State, and Country Please.
            (3) When was the company founded? 
            
            Please search for these answers from their official website or reliable sources. Please create a search that would likely search their own content. 
            """
    
    response_format = """
        Please provide your answer in the following format:
        {
            "Company Name": "Your concise answer here",
            "Full Time Employees": "The Number of Full Time Employees",
            "Location": "The Location Based on the information you found",
            "Founding Year": "The Founding Year of the company",
            "sources": ["list", "of", "sources", "used"]
        }
        
        e.g.
        {
            "Company Name": "Company ABC",
            "Full Time Employees": "150", 
            "Location": "Berlin, Berlin, Germany",
            "Founding Year": "1960",
            "sources": ["www.companya.com", "www.wikipedia.com"]
        }
        
        ANSWER TYPES:
        - Full Time Employees: A number (e.g., 150)
        NOTE: Do not provide ranges or modifying words. Only say a number. Adding any extra context will break the response.
        - Location: City, State, and Country (e.g., Berlin, Berlin, Germany)
        - Founding Year: A year (e.g., 1960)
        
        IMPORTANT: DO NOT MAKE UP INFORMATION. ONLY PROVIDE ANSWERS BASED ON THE INFORMATION YOU FIND IN THE SEARCH RESULTS.
        PLEASE DO NOT ELABORATE ON WHY YOU CHOSE ANY ANSWER. JUST PROVIDE THE RAW INFORMATION. 
        """
    
    for row in tqdm(rows, desc="Processing rows", unit="row"):
        input_file_information = f"""
            Company Name: {row['CompanyName']}
            Description: {row['CompanyDescription']}
            """
        
        query_prompt = f"""
            Based on the following information, generate a search query to find information that would help answer the question in the prompt.

            {input_file_information}

            Prompt: {prompt}
            
            NOTE: PLEASE ONLY RESPOND IN THIS FORMAT. ADDITIONAL TEXT WILL CAUSE OUR QUERY TO FAIL.
            
            {{
                "query": "Your generated search query here"
            }}
            

            Generate a search query:
            """
        
        result = process_csv_row(row, prompt, query_prompt, input_file_information, response_format)
        results.append(result)
    
    write_json_output(results, output_file)

if __name__ == "__main__":
    input_file = "input/test.csv"
    output_file = "output/output3.json"
    main(input_file, output_file)