from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.chat_backend import (
    ChatRequest,
    get_chat_model_list,
    handle_chat,
    init_model_and_tokenizer,
)
from utils.embedding_backend import EmbeddingRequest, handle_embedding


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model_and_tokenizer()
    yield


app = FastAPI(lifespan=lifespan)

app = FastAPI()


class ModelsResponse(BaseModel):
    models: list[str]


@app.get("/api/model-list")
def model_list():
    return ModelsResponse(models=[*get_chat_model_list()])


@app.get("/api/chat")
def chat(req: ChatRequest):
    response = handle_chat(req)
    return StreamingResponse(response)


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/api/embed")
def embed(req: EmbeddingRequest):
    response = handle_embedding(req)
    return EmbeddingResponse(embeddings=response)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    class Args(BaseModel):
        host: str
        port: int

    args = Args.model_validate(parser.parse_args().__dict__)

    uvicorn.run(app, host=args.host, port=args.port)
