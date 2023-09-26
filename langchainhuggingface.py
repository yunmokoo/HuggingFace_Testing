import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain.schema import HumanMessage, BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

os.environ["HF_HOME"] = ""
os.environ["HF_TOKEN"] = ""

# Use the model and tokenizer as needed for your task.

# Set the default data type to 'torch.float32' (single-precision)
torch.set_default_dtype(torch.float32)

model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Use the model and tokenizer as needed for your task.

input_text = '''Talk like a terran race in Starcraft 1 broodwar'''

# Encode the input text
inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False)

# Ensure the input is of the correct data type ('Long')
inputs = {k: v.to(torch.long) for k, v in inputs.items()}

class CommaSeparate(BaseOutputParser):
    '''Parse output of an LLM call to CommaSeparated list'''
    def parse(self, text: str):
        '''Parse output'''
        return text.strip().split(", ")


outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
human_template = text

# Define the conversation using tuples
conversation = [
    ('system', input_text),
    ('human', human_template),  # You may need to define 'human_template' here
]

chat_prompt = ChatPromptTemplate.from_messages(conversation)



chain = chat_prompt | model() | CommaSeparate()
result = chain.invoke({'text': 'World leaders'})
print(result)