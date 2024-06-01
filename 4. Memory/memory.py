# --- Working with memory: Vanilla ---
#  Implementing a vanilla memory system to store and retrieve information from previous interactions.
#
# Author: Naomi Lago
# Date: 2024-05-31

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from config.utils import *
import os

os.system('clear')
set_debug(True)

llm = ChatOpenAI(
  model='gpt-3.5-turbo-0125',
  temperature=0.6
)

messages = [
  'I\'m looking for a book to read. Can you help me?',
  'Why is important to read books?',
  'Do you have any tips for reading without getting bored?',
  'In my case, I often find myself getting distracted when I read. What can I do to avoid this?'
]

long_message = ''
for message in messages:
  long_message += f'User: {message}\n'
  long_message += f'AI: '
  
  template = PromptTemplate(template=long_message, input_variables=[''])
  chain = template | llm | StrOutputParser()
  
  res = chain.invoke({})
  logger.success(res)
  
  long_message += res + '\n'