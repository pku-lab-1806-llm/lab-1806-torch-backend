from threading import Thread
from typing import Any, Iterator, Literal, Union

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer

if torch.__version__.endswith("cu121"):
    print("Using CUDA 12.1 (for mock)")
    device = "cuda:0"  # the device to load the model onto
    model_name = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
    global_chat_model = "Qwen2.5-0.5B-Instruct"
else:
    print("Using NPU")
    device = "npu"  # the device to load the model onto
    model_name = "./models/Qwen2.5-7B-Instruct"
    global_chat_model = "Qwen2.5-7B-Instruct"

global_model: Union[Qwen2ForCausalLM, None] = None
global_tokenizer: Union[Qwen2Tokenizer, None] = None


def get_chat_model_list():
    return [global_chat_model]


def init_model_and_tokenizer():
    global global_model, global_tokenizer
    global_model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    global_tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
    print("Chat Model and tokenizer initialized")


def get_global_model_and_tokenizer():
    if global_model is None or global_tokenizer is None:
        init_model_and_tokenizer()
    assert global_model is not None, "Model is not initialized"
    assert global_tokenizer is not None, "Tokenizer is not initialized"
    return global_model, global_tokenizer


def as_auto_tokenizer(tokenizer: Qwen2Tokenizer) -> AutoTokenizer:
    any_tokenizer: Any = tokenizer
    return any_tokenizer


class ChatMsg(BaseModel):
    """Message in chat request."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
        }

    @classmethod
    def to_dict_list(cls, messages: "list[ChatMsg]"):
        return [msg.to_dict() for msg in messages]


class ModelOptions(BaseModel):
    max_tokens: Union[int, None] = None
    temperature: Union[float, None] = None


class ChatRequest(BaseModel):
    # empty string for default model
    model: str
    messages: list[ChatMsg]
    options: Union[ModelOptions, None] = None


def handle_chat(req: ChatRequest) -> Iterator[str]:
    model, tokenizer = get_global_model_and_tokenizer()
    assert req.model == global_chat_model, f"Invalid model: {req.model}"
    text = tokenizer.apply_chat_template(
        ChatMsg.to_dict_list(req.messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(text, str), "Failed to apply chat template"
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 创建流式输出器
    streamer = TextIteratorStreamer(
        as_auto_tokenizer(tokenizer),
        skip_prompt=True,
        skip_special_tokens=True,
    )
    model.generate

    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

    thread.join()


if __name__ == "__main__":
    init_model_and_tokenizer()

    response = handle_chat(
        ChatRequest(
            model=global_chat_model,
            messages=[ChatMsg(role="user", content="中英文各一句话介绍LLM")],
        )
    )
    for msg in response:
        print(msg, end="")
