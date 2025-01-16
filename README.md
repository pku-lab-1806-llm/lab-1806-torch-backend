# lab-1806-torch-backend

A simple inference backend with PyTorch for `lab-1806-rag` repo.

Compatible with both CUDA and Ascend.

## Usage

Develop With CUDA:

```bash
# > dev
uv run -- uvicorn main:app --port 8011 --reload
```

Run With Ascend:

- Install torch, torch_npu, transformers by yourself or with existing images.
- Install fastapi and uvicorn which may not be contained in general images.

  ```bash
  pip install -r requirements.txt
  ```

- Put the weights and config files of `Qwen2.5-7B-Instruct` at `./models/Qwen2.5-7B-Instruct` or `$SCOW_AI_MODEL_PATH`.
- Put the weights and config files of `sentence-transformers/all-MiniLM-L6-v2` in `../data/sentence-transformers/all-MiniLM-L6-v2`.
- Start the server:

  ```bash
  # > prod
  python main.py
  ```
