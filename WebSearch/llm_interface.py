import anthropic
import json

from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_search_query(row, prompt):
    query_prompt = f"""
    Based on the following information about a company, generate a search query to find information that would help answer the question in the prompt.

    Company Name: {row['company_name']}
    Company URL: {row['company_url']}
    Description: {row['description']}

    Prompt: {prompt}

    Generate a search query:
    """

    response = client.completions.create(
        model="claude-3-haiku-20240307",
        prompt=query_prompt,
        max_tokens_to_sample=100,
        temperature=0.5
    )

    return response.completion.strip()

def generate_answer(row, page_contents, prompt):
    answer_prompt = f"""
    Based on the following information about a company and the content from relevant web pages, please answer the question in the prompt.

    Company Name: {row['company_name']}
    Company URL: {row['company_url']}
    Description: {row['description']}

    Prompt: {prompt}

    Relevant web page contents:
    {json.dumps(page_contents, indent=2)}

    Please provide your answer in the following JSON format:
    {{
        "answer": "Your concise answer here",
        "explanation": "A brief explanation of your answer",
        "sources": ["list", "of", "sources", "used"]
    }}
    """

    response = client.completions.create(
        model="claude-3-haiku-20240307",
        prompt=answer_prompt,
        max_tokens_to_sample=1000,
        temperature=0.1
    )

    return json.loads(response.completion)