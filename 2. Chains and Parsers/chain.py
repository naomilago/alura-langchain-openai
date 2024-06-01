# --- Working with Chains ---
# Chains are a way to create a sequence of operations that will be executed in order.
# - Each operation in the chain can be a prompt, a parser, or another chain.
#
# Project: Travel Suggestion
# Author: Naomi Lago
# Date: 2024-05-30

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from config.utils import *
from typing import *

os.system('clear')

city_template = ChatPromptTemplate.from_template('''
  - Você vai sugerir uma cidade, no Brasil, com base em meu interesse em {interest}.
  - Você irá fornecer clareza no(s) motivo(s) pelos quais eu deveria visitar a cidade com base em meu interesse.
  - Você terá bastante liberdade para sugerir a cidade brasileira que achar mais adequada, mas lembre-se:
    - Sua resposta deve ser clara e objetiva.
    - Você deve deixar claro o motivo pelo qual a cidade se alinha com meu interesse.
''')

restaurant_template = ChatPromptTemplate.from_template('''
  Considere a sugestão de cidade abaixo:
  
  {city}
              
  - Você vai sugerir um restaurante baseado na cidade acima.
  - Você irá fornecer clareza no(s) motivo(s) pelos quais eu deveria visitar o restaurante.
  - Seus motivos devem deixar clara a localização do restaurante e o motivo pelo qual ele é especial.
  - Sua resposta deve conter, obrigatoriamente, os seguintes itens:
    - Nome do restaurante
    - Localização do restaurante
    - Motivo pelo qual o restaurante é especial
    - Como se alinha com meu interesse
''')
                                                        
activity_template = ChatPromptTemplate.from_template('''
  Considere as sugestões de cidade e restaurante abaixo:

  {restaurant}
                                                     
  - Você irá sugerir uma atividade para fazer na cidade, baseada no meu interesse.
  - Você irá fornecer clareza no(s) motivo(s) pelos quais eu deveria fazer a atividade - que será a atividade principal da viagem.
  - Sua resposta deve conter, obrigatoriamente, os seguintes itens:
    - Nome da atividade
    - Localização da atividade
    - Motivo pelo qual a atividade é especial
    - Como se alinha com meu interesse
  - Sua resposta será no formato conversacional, contendo as informações acima, em um formato de conversa de amigos.
''')

llm = ChatOpenAI(
  model_name='gpt-3.5-turbo',
  temperature=0.5,
)

@chain
def travel_chain(interest: str) -> Any:
  '''Suggests a city, restaurant and activities based on user input.'''
  
  city_chain = city_template | llm | StrOutputParser()
  restaurant_chain = restaurant_template | llm | StrOutputParser()
  activity_chain = activity_template | llm | StrOutputParser()
  
  res = city_chain.invoke(input={'interest': interest})
  res = restaurant_chain.invoke(input={'city': res})
  res = activity_chain.invoke(input={'restaurant': res})
  
  return res

print(travel_chain.invoke('Eu gosto muito de processamento de linguagem natural'))