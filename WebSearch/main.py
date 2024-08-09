# main.py
from search_engine import get_search_results, get_page_content
from claude_interface import generate_search_query, generate_answer
from csv_handler import read_csv, write_json_output
from config import BRAVE_API_KEY

def process_csv_row(row, prompt):
    # Generate search query
    query = generate_search_query(row, prompt)
    
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
    answer = generate_answer(row, page_contents, prompt)
    
    return answer

def main(input_file, output_file, prompt):
    rows = read_csv(input_file)
    results = []
    
    for row in rows:
        result = process_csv_row(row, prompt)
        results.append(result)
    
    write_json_output(results, output_file)

if __name__ == "__main__":
    input_file = "input.csv"
    output_file = "output.json"
    prompt = "What are the main products or services offered by this company?"  # Example prompt
    main(input_file, output_file, prompt)