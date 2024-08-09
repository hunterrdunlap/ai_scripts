import anthropic
import json
from typing import Dict, List, Any

from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_search_query(row: Dict[str, str], prompt: str, query_prompt: str) -> str:
    
    print("searching web...")
    response = client.completions.create(
        model="claude-3-haiku-20240307",
        prompt=query_prompt,
        max_tokens_to_sample=100,
        temperature=0.1
    )

    return response.completion.strip()

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

    response = client.completions.create(
        model="claude-3-haiku-20240307",
        prompt=answer_prompt,
        max_tokens_to_sample=1000,
        temperature=0.1
    )

    return json.loads(response.completion)