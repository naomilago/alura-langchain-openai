# --- Working with memory: Graph ---
# Implementing the ConversationKGMemory to store and retrieve information from previous interactions.
# - The KG memory stores the triples of the knowledge graph, which can be accessed by the chain to retrieve information from previous interactions.
#
# Author: Naomi Lago
# Date: 2024-05-31

import sys
sys.path.append('D:\Large Langage Models\LangChain\Alura Course')

from langchain.memory import ConversationKGMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
import matplotlib.pyplot as plt
from config.utils import *
import networkx as nx
import os

os.system('clear')
set_debug(True)

llm = ChatOpenAI(
  model='gpt-3.5-turbo-0125',
  temperature=0.6
)

template = '''
The following, is a conversation between a Human and an AI. The AI is talkative and always try to support the Human with the best answers.
The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.
If the AI does not know the answer, it will truthfully say it doesn't know.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:
'''

prompt = PromptTemplate(
  template=template,
  input_variables=['history', 'input']
)

conversation = ConversationChain(
  llm=llm,
  verbose=True,
  prompt=prompt,
  memory=ConversationKGMemory(llm=llm)
)

conversation.predict(input='How is the conversational GenAI popularity related to the inherit human need for companionship that other humans aren\'t being able to provide anymore? Answer with clear, concise, and logical arguments.')

print(triples := conversation.memory.kg.get_triples())

G = nx.DiGraph()

plt.figure(figsize=(12, 8))
plt.rcParams['savefig.dpi'] = 300

for s, p, o in triples:
  G.add_edge(s, o, label=p)

pos = nx.spring_layout(G)
edge_labels = nx.get_edge_attributes(G, 'label')

nx.draw(G, pos, with_labels=True, node_size=1500, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)  

plt.savefig('./4. Memory/graph.png', dpi=300, bbox_inches='tight')
plt.close()