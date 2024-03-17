# to run
# python main.py --language javascript --task 'print hello'
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="retuan a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = Ollama(model="gemma:2b")

# result = llm.invoke("write a very very short poem")
# print(result)

code_prompt = PromptTemplate(
  template="Write a very short {language} function that will {task}",
  input_variables=["language", "task"]
)

code_chain = code_prompt | llm

result = code_chain.invoke({
  "language": args.language,
  "task": args.task
})
print(result)