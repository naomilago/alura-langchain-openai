from loguru import logger
import tiktoken
import json
import os

def tokens_counter(string: str, encoding_name: str = 'cl100k_base') -> int:
    '''Returns the number of tokens in a text string.'''
    
    encoding = tiktoken.get_encoding(encoding_name)    
    return len(encoding.encode(string))

def json_viewer(path: dict, clear: bool = True) -> None:
    '''Prints a JSON object in a human-readable format.'''

    if clear: os.system('clear')

    with open(path, 'r') as f:
        output = json.load(f)
        f.close()

    print('\n', json.dumps(output, indent=2), '\n')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = f'OpenAI Usage'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')