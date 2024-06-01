# --- Working with Information Retrieval: *********** ---
# Implementing a simple information retrieval system using the *********** class to load multiply .pdf files and split them into chunks of text.
# 
# Project: A mental heath company that answers questions about mental health in a friendly way.
# Author: Naomi Lago
# Date: 2024-05-31

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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

loaders = [
  PyPDFLoader('./5. Retrieval/data/Mental-Health-Resources---DEI.pdf'),
  PyPDFLoader('./5. Retrieval/data/Soc 426 Gender & mental health.pdf'),
  PyPDFLoader('./5. Retrieval/data/Spreading-Mental-Health.pdf')
]

documents = []
for doc in loaders:
  documents.extend(doc.load())

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
  documents=texts,
  embedding=embeddings
)

template = '''
- You are an empathic mental health coach answering questions about mental health in a very friendly way. 
- If you don't know the answer, say you don't
- If the user input is out of scope, say it's out of scope and you can't help. 
- If the user input is offensive, say you don't tolerate offensive language.
- Your answer will be respectful, clear, concise, friendly, and educative.
- If applicable, provide resources or references to help the user.

Use the following CONTEXT to help you formulate the answer, but always give more importance to the top contexts to provide the best answer possible.:

{context}

Now, answer the following USER QUESTION below:

{question}
'''

chain = RetrievalQA.from_chain_type(
  llm=llm,
  retriever=db.as_retriever(search_kwargs={'k': 5}),
  chain_type_kwargs={
    'prompt': PromptTemplate(template=template, input_variables=['query', 'context'])
  }
)

query = '''
I'm a black transgender person, how could I find mental health resources that are inclusive and respectful of my identity?
'''

res = chain.invoke({'query': query})