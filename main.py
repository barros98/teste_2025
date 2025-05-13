from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Modelo de dados esperado
class Usuario(BaseModel):
    nome: str
    idade: int
    email: str

# Rota POST para cadastrar usuário
@app.post("/usuarios/")
def criar_usuario(usuario: Usuario):
    return {
        "mensagem": f"Usuário {usuario.nome} criado com sucesso!",
        "dados": usuario.dict()
    }
