# Web Search and Answer Generation Project

This project uses the Brave search API and Claude AI to process questions from a CSV file, search for relevant information, and generate answers in JSON format.

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up your API keys in `config.py`:
   - ANTHROPIC_API_KEY: Your Claude API key
   - BRAVE_API_KEY: Your Brave Search API key
   - GOOGLE_API_KEY: Your Google API key

3. Prepare your input CSV file with at least a 'question' column.

## Usage

Run the main script:

```
python main.py
```

This will process the input CSV file and generate an output JSON file with answers.

## File Structure

- `main.py`: Main script to run the project
- `search_engine.py`: Functions for generating search queries and fetching results
- `llm_interface.py`: Interface with Claude AI for answer generation
- `csv_handler.py`: Functions for reading CSV and writing JSON output
- `config.py`: Configuration file for API keys
- `requirements.txt`: List of required Python packages