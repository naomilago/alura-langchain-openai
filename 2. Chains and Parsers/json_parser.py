# --- Define the JSON Output Parser ---
# The JSON Output Parser is a class that will be used to parse the output of the LLM into a JSON format. 
# - This is useful when you want to return structured data from the LLM, such as a list of recommendations or a list of steps in a process. 
#
# Project made: Study Roadmap Generator
# Author: Naomi Lago
# Date: 2024-05-31

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from config.utils import *
from typing import *

from langchain_core.output_parsers import (
  JsonOutputParser,
  StrOutputParser,
)

from langchain_core.pydantic_v1 import (
  BaseModel,
  Field
)

os.system('clear')
set_debug(True)

class StatusField(BaseModel):
  category: str = Field(..., description='The category, overall, of the response. If the input is out of scope, the value is "out of scope".')
  message: str = Field(..., description='A message explaining in a very short way, the abstract of the response. If the input is out of scope, you must return why.')

class RoadmapField(BaseModel):
  step: str = Field(..., description='The step in the roadmap, with a short title that align with the purpose.')
  description: str = Field(..., description='The description of the step in the  current roadmap item, with a very short explanation of the reasons why it is important.')
  
class OutputField(BaseModel):
  status: StatusField = Field(..., description='The status of the response, with the category and message.')
  subject: str = Field(..., description='The subject suggested to study, based on the user interest - written in few words.')
  reason: str = Field(..., description='The reason why the subject aligns with the user interest - written in human-like language.')
  roadmap: List[RoadmapField] = Field(..., description='The roadmap for studying the subject, with a clear path from starting to mastering the subject.')

parser = JsonOutputParser(pydantic_object=OutputField)

template = '''
  - You'll suggest me a new subject to study, in technology, based the user interest.
  - You'll provide clarity on the reason(s) why I should study the subject based on my interest.
  - You'll have plenty of freedom to suggest the technology subject you find most suitable, but remember:
    - Your answer must be clear and objective.
    - You must make clear the reason why the subject aligns with my interest.
  - You'll suggest a concise and clear study roadmap, with a clear path since starting until mastering the subject.
  - You're answer will be matching the language the user have inputed - to be more natural and coherent.
    - For example: If the user input for the interest is in Portuguese, you must answer in Portuguese.
  - You'll always adhere to the rules above, so in case of the user input is out of scope, you must only fill the status field and leave the others with empty strings.

  USER INPUT: {interest}

  OUTPUT TEMPLATE FORMAT:
  {output}
'''

study_template = PromptTemplate(
  template=template,
  input_variables=['interest'],
  partial_variables={'output': parser.get_format_instructions()}
)

llm = ChatOpenAI(
  model_name='gpt-3.5-turbo-0125',
  temperature=0.5,
)

study_chain = study_template | llm | JsonOutputParser()

res = study_chain.invoke('Quero estudar UX/UI.')

with open('./2. Chains and Parsers/output.json', 'w') as f:
  f.write(str(res).replace('\'', '"'))
  f.close()
  
json_viewer('./2. Chains and Parsers/output.json')