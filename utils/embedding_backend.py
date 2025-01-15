from typing import Union

import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

if torch.__version__.endswith("cu121"):
    default_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
else:
    default_embedding_model_name = "../data/sentence-transformers/all-MiniLM-L6-v2"

default_embedding_model: Union[SentenceTransformer, None] = None


def init_embedding_model():
    global default_embedding_model
    default_embedding_model = SentenceTransformer(default_embedding_model_name)
    print("Embedding model initialized")


def get_embedding_model():
    if default_embedding_model is None:
        init_embedding_model()
    assert default_embedding_model is not None, "Model is not initialized"
    return default_embedding_model


def get_embedding_model_list():
    return [default_embedding_model_name]


class EmbeddingRequest(BaseModel):
    model: str
    texts: list[str]


def handle_embedding(req: EmbeddingRequest) -> list[list[float]]:
    assert req.model == default_embedding_model_name, f"Unsupported model: {req.model}"
    return get_embedding_model().encode(req.texts).tolist()


if __name__ == "__main__":
    init_embedding_model()

    response = handle_embedding(
        EmbeddingRequest(model=default_embedding_model_name, texts=["Hello, world!"])
    )
    print(
        len(response[0]),
        response[0][:5],
    )
