import os

# LangChain
import langchain
print(langchain.__version__)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

# Pydantic
from typing import List
from pydantic import BaseModel, Field


# Preparando o ambiente
load_dotenv(find_dotenv())
os.environ['LANGSMITH_API_KEY'] = str(os.getenv('LANGSMITH_API_KEY'))
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'default'

# Fazendo uma chamada
llm = ChatOpenAI(temperature = 0, model = 'gpt-4o-mini')
response = llm.invoke('Quando o Brasil foi descoberto?')
response.content

# Parsing Output
output_parser = StrOutputParser()
output_parser.invoke(response)

# Simple Chain
chain = llm | output_parser
response2 = chain.invoke('Quem descobriu o brasil?')
print(response2)

