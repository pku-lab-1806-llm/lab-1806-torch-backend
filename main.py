from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel

from utils.embedding_backend import (
    EmbeddingRequest,
    get_embedding_model_list,
    handle_embedding,
    init_embedding_model,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = app
    init_embedding_model()
    yield


app = FastAPI(lifespan=lifespan)


class ModelsResponse(BaseModel):
    models: list[str]


@app.get("/api/model-list")
def model_list():
    return ModelsResponse(models=[*get_embedding_model_list()])


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


@app.post("/api/embed")
def embed(req: EmbeddingRequest):
    response = handle_embedding(req)
    return EmbeddingResponse(embeddings=response)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)

    class Args(BaseModel):
        host: str
        port: int

    args = Args.model_validate(parser.parse_args().__dict__)

    uvicorn.run(app, host=args.host, port=args.port)
