from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

default_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
default_embedding_model = SentenceTransformer(default_embedding_model_name)


def get_embedding_model_list():
    return [default_embedding_model_name]


class EmbeddingRequest(BaseModel):
    model: str
    texts: list[str]


def handle_embedding(req: EmbeddingRequest) -> list[list[float]]:
    assert req.model == default_embedding_model_name, f"Unsupported model: {req.model}"
    return default_embedding_model.batch_encode_text(req.texts)
