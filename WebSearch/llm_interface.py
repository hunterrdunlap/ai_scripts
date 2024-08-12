import anthropic
import json
from typing import Dict, List, Any

from config import ANTHROPIC_API_KEY, MODEL

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_search_query(row: Dict[str, str], prompt: str, query_prompt: str) -> str:
    
    print("searching web for information on  ", row["CompanyName"])    
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": query_prompt}
        ]
    )
    
    try:
        response = message.content[0].text
    except:
        ValueError("No response from Claude") 
        
    # Parse the JSON response
    response_json = json.loads(response)
    
    # Extract the query
    return response_json["query"]

def generate_answer(row: Dict[str, str], page_contents: List[Dict[str, str]], prompt: str, input_file_information: str, response_format: str) -> Dict[str, str]:
    print("generating answer...")
    answer_prompt = f"""
    Based on the following information and the content from relevant web pages, please answer the question in the prompt.

    {input_file_information}

    Prompt: {prompt}

    Relevant web page contents:
    {json.dumps(page_contents, indent=2)}

    {response_format}
    """

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.1,
        messages=[
            {"role": "user", "content": answer_prompt}
        ]
    )

    try:
        response = message.content[0].text
        return json.loads(response)
    except json.JSONDecodeError:
        print("Warning: Response is not in JSON format. Returning raw text.")
        return {"raw_response": response}
    except IndexError:
        raise ValueError("No response from Claude")