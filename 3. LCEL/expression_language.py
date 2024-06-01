# --- Define the LCEL (Langchain Language Expression) ---
# The LCEL is a simple language that will be used to define the structure of the conversation.
# - This is useful when you want to define a conversation structure in a simple and readable way.
#
# Project made: Book recommendation
# Author: Naomi Lago
# Date: 2024-05-31

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from config.utils import *
import os

from langchain_core.output_parsers import (
  JsonOutputParser,
  StrOutputParser
)

from langchain_core.pydantic_v1 import (
  BaseModel,
  Field
)

from langchain.prompts import (
  ChatPromptTemplate,
  PromptTemplate
)

os.system('clear')
set_debug(True)

llm = ChatOpenAI(
  model='gpt-3.5-turbo-0125',
  temperature=0.6
)

template = PromptTemplate.from_template('''
You are an experience librarian that will recommend a book to a user based on their interest. You must provide:
- The title of the book.
- The author of the book.
- The genre of the book.
- The year the book was published.
- A brief summary of the book.
- The reason why you are recommending this book.

Your answer must be clear and objective, and you must always adhere to the rules above.
Your answer will be in a conversational tone, matching the language the user has inputted.
In case the user input is out of scope, you will reply kindly that you can't help.

Here's the information I need to recommend to the user in a conversational tone:

Title: {title}
Author: {author}
Genre: {genre}
Year: {year}
Summary: {summary}
Reason: {reason}

Oh, and you will always match the language the user has inputted. For example, if the user input is in Portuguese, you must answer in Portuguese.
''')

class RecommendationFields(BaseModel):
  title: str = Field(..., description='The title of the book.')
  author: str = Field(..., description='The author of the book.')
  genre: str = Field(..., description='The genre of the book.')
  year: int = Field(..., description='The year the book was published.')
  summary: str = Field(..., description='A brief summary of the book.')
  reason: str = Field(..., description='The reason why you are recommending this book.')

parser = JsonOutputParser(pydantic_object=RecommendationFields)

recommendation_template = PromptTemplate(
  template='Recommend a book to the user based on their interest: {interest}. {output}',
  input_variables=['interest'],
  partial_variables={'output': parser.get_format_instructions()}
)

recommendation_chain = recommendation_template | llm | parser
main_chain = template | llm | StrOutputParser()

chain = recommendation_chain | main_chain

print('\n\n', chain.invoke('Quero um livro que eu, que nao consigo focar nas leituras que comeco, possa terminar.'), '\n\n')