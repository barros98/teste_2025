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



# --- Similarity Search ----------------------------------------------------------------------------------------------------------------------------------------------------------
query = 'Qual o objetivo do Componente Pd?'
search_results = vector_store.similarity_search(query, 1)
print(search_results[0].page_content)
# Dessa forma não conseguimos colocar esse processo em produção com LCEL do langchain. Precisamos de um retriever.



# --- Encontrando os melhores documentos com Retrieve ----------------------------------------------------------------------------------------------------------------------------------------------------------
query = 'Qual o objetivo do Componente Pd?'
retriever = vector_store.as_retriever(search_kwargs = {'k':3})
retriever.invoke(query)



# --- Prompt Template ----------------------------------------------------------------------------------------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate

# Definindo o template
template = """Responda a pergunta a seguir se baseando apenas no seguinte contexto:
{context} 

Pergunta:
{question}

Resposta:
"""

# Definindo o prompt template
prompt = ChatPromptTemplate.from_template(template)



# --- RAG Chain ----------------------------------------------------------------------------------------------------------------------------------------------------------
from langchain.schema.runnable import RunnablePassthrough

# Função que transforma os retrievers em um único texto
def docs2str(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# Chain
rag_chain = ({'context': retriever | docs2str, 'question': RunnablePassthrough()}
             | prompt
             | ChatOpenAI(temperature = 0, model = 'gpt-4o-mini')
             | StrOutputParser()
             )
response = rag_chain.invoke(query)
print(response)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Conversational RAG ----------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Chat History ----------------------------------------------------------------------------------------------------------------------------------------------------------
from langchain_core.messages import HumanMessage, AIMessage
chat_history = []
chat_history.extend(
    HumanMessage(content = query),
    AIMessage(content = response)
    )





from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 

contextualize_q_system_prompt = """
Dado um histórico de conversa e a pergunta mais recente do usuário, a qual pode fazer referência a esse histórico, 
formule uma pergunta independente que possa ser compreendida sem a necessidade do histórico da conversa. NÃO responda à pergunta,
apenas reformule-a se necessário ou retorne-a como está.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
print(contextualize_chain.invoke({"input": "Como esse componente é utilizado?", "chat_history": chat_history}))









history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


from langchain_core.messages import HumanMessage, AIMessage

chat_history = []
question1 = 'Qual o objetivo do Componente Pd?'
answer1 = rag_chain.invoke({"input": question1, "chat_history": chat_history})['answer']
chat_history.extend([
    HumanMessage(content=question1),
    AIMessage(content=answer1)
])

print(f"Human: {question1}")
print(f"AI: {answer1}\n")

question2 = "Como esse componente é utilizado?"
answer2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})['answer']
chat_history.extend([
    HumanMessage(content=question2),
    AIMessage(content=answer2)
])

print(f"Human: {question2}")
print(f"AI: {answer2}")


# --- Setting Up the SQLite Database ----------------------------------------------------------------------------------------------------------------------------------------------------------
import sqlite3
from datetime import datetime
import uuid

DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_query TEXT,
    gpt_response TEXT,
    model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

# Initialize the database
create_application_logs()


# Example usage for a new user
session_id = str(uuid.uuid4())
question = 'Qual o objetivo do Componente Pd?'
chat_history = get_chat_history(session_id)
answer = rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']
insert_application_logs(session_id, question, answer, 'gpt-4o-mini')
print(f"Human: {question}")
print(f"AI: {answer}\n")

# Example of a follow-up question
question2 = "O que esse componente faz?"
chat_history = get_chat_history(session_id)
answer2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})['answer']
insert_application_logs(session_id, question2, answer2, 'gpt-4o-mini')
print(f"Human: {question2}")
print(f"AI: {answer2}")



