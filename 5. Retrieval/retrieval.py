# --- Working with Information Retrieval: TextLoader ---
# Implementing a simple information retrieval system using the TextLoader class to load a text file and split it into chunks of text.
# - The TextLoader class is used to load a text file, which we can split into chuncks later.
# 
# Project: An Atheist Scientist that answers questions in a friendly way.
# Author: Naomi Lago
# Date: 2024-05-31

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
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

loader = TextLoader(file_path='./5. Retrieval/Atheism - An Affirmative View (1980).txt')
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
  documents=texts,
  embedding=embeddings
)

template = '''
- You are an atheist scientist answering questions about this religion in a very friendly way.
- If you don't know the answer, say you don't; if the user input is out of scope, say it's out of scope and you can't help.
- If the user input is offensive, say you don't tolerate offensive language.
- Your answer will be clear, concise, friendly, and educative.

Use the following CONTEXT to help you formulate the answer, but always give more importance to the top contexts:

{context}

Now, answer the following USER QUESTION below:

{question}
'''

chain = RetrievalQA.from_chain_type(
  llm=llm,
  retriever=db.as_retriever(search_kwargs={'k': 2}),
  chain_type_kwargs={
    'prompt': PromptTemplate(template=template, input_variables=['query', 'context'])
  }
)

query = '''
What are intelligent people working for?
'''

res = chain.invoke({'query': query})