from fastapi import FastAPI

# Cria a aplicação FastAPI
app = FastAPI()

# Rota simples de teste
@app.get("/")
def read_root():
    return {"mensagem": "Olá, FastAPI!"}

# Rota com parâmetro na URL
@app.get("/saudacao/{nome}")
def saudacao(nome: str):
    return {"mensagem": f"Olá, {nome}!"}

# Rota com query parameter
@app.get("/soma")
def somar(a: int, b: int):
    return {"resultado": a + b}
