# --- Hello World, the first building blocks ---
# The first step in any project is to make sure everything is set up correctly and make it exciting.
# 
# Project made: Family Itinerary
# Author: Naomi Lago
# Date: 2024-05-30 

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config.utils import *
import os

os.system('clear')

days = 2
kids = 3
place = 'park'
activity = 'water activities in the lake'

template = PromptTemplate.from_template('''
  Create a {days} day itinerary for a family with {kids} kids. 
  They want to spend most of their time at the {place} doing {activity}.
''')

prompt = template.format(
  activity=activity,
  place=place,
  days=days,
  kids=kids
)

prompt_tokens = tokens_counter(prompt)

llm = ChatOpenAI(
  model_name='gpt-3.5-turbo',
  temperature=0.6,
  max_tokens=500,
)

response = llm.invoke(prompt).content
response_tokens = tokens_counter(response)

print('\n', response)

print('-'*60)
print(f'\nTotal tokens: {prompt_tokens} (prompt) + {response_tokens} (response) = {prompt_tokens + response_tokens} tokens\n')