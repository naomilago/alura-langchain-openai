# --- Working with memory: Summary ---
# Implementing the ConversationSummaryMemory to store and retrieve information from previous interactions.
# - The summary memory is a memory that stores the last k messages of the conversation and summarizes the rest.
#
# Author: Naomi Lago
# Date: 2024-05-31

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
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

memory = ConversationSummaryMemory(
  k=2,
  llm=llm,
)

conversation = ConversationChain(
  llm=llm,
  verbose=True,
  memory=memory
)

for message in messages:
  res = conversation.predict(input=message)
  logger.info(res)
  
logger.success(conversation.predict(input='What was the second thing I asked?'))