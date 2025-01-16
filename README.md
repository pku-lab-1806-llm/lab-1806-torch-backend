# lab-1806-torch-backend

A simple inference backend with PyTorch for `lab-1806-rag` repo.

Compatible with both CUDA and NPU.

## Usage

```bash
# > dev
uv run -- uvicorn main:app --port 8011 --reload

# > prod
uv run main.py
```

## Together With NPU

Put the weights and config files of `Qwen2.5-7B-Instruct` at `./models/Qwen2.5-7B-Instruct` or `$SCOW_AI_MODEL_PATH`.
Put the weights and config files of `sentence-transformers/all-MiniLM-L6-v2` in `../data/sentence-transformers/all-MiniLM-L6-v2`.
