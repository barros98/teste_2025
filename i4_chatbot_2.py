# Base
import os
from dotenv import load_dotenv, find_dotenv

# LangChain
import langchain
print(langchain.__version__)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


from langchain_openai import OpenAIEmbeddings

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

# --- Criando o prompt template ----------------------------------------------------------------------------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate



# --- Carregando e dividindo o arquivo ----------------------------------------------------------------------------------------------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

pdf_loader = PyPDFLoader('areh20233181_1.pdf')
pdf = pdf_loader.load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
pdf_split = text_spliter.split_documents(pdf)

len(pdf)
len(pdf_split)


# --- Embedding ----------------------------------------------------------------------------------------------------------------------------------------------------------
embeddings = OpenAIEmbeddings()
doc_embed = embeddings.embed_documents([split.page_content for split in pdf_split])


# --- Vector database ----------------------------------------------------------------------------------------------------------------------------------------------------------
from langchain_chroma import Chroma
vector_store = Chroma.from_documents(
    collection_name = 'vector_database_1',
    documents = pdf_split,
    embedding = embeddings,
    persist_directory = 'C:\\Users\\vitor\\OneDrive\\i4\\i4 Agent'
    )












