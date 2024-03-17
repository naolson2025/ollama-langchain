# to run
# python main.py --language javascript --task 'print hello'
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = Ollama(model="gemma:2b")

# basic run with this line of code
# result = llm.invoke("write a very very short poem")
# print(result)

code_prompt = PromptTemplate(
  input_variables=["language", "task"],
  template="Write a very short {language} function that will {task}"
)

test_prompt = PromptTemplate(
  input_variables=["language", "code"],
  template="Write a test for the followin {language} code: \n{code}"
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt,
  output_key="code"
)

test_chain = LLMChain(
  llm=llm,
  prompt=test_prompt,
  output_key="test"
)

# chain the output from our first chain to the input of the second chain
# the LLM will use the output from the first prompt as input in the second prompt
chain = SequentialChain(
  chains=[code_chain, test_chain],
  input_variables=["language", "task"],
  output_variables=["code", "test"]
)

result = chain.invoke({
  "language": args.language,
  "task": args.task
})

print(">>>>>>GENERATED CODE<<<<<<")
print(result["code"])

print(">>>>>>GENERATED TEST<<<<<<")
print(result["test"])
