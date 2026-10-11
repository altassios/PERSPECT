from typing import Optional


class ChatModel:

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError


class HFChatModel(ChatModel):
    def __init__(self, model_name: str, temperature: float, top_p: float, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.torch = torch
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def _apply_template(self, messages):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def chat(self, user_prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        text = self._apply_template(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with self.torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OpenAICompatibleChatModel(ChatModel):
    def __init__(self, model_name, base_url, temperature, top_p, max_new_tokens, api_key="EMPTY"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def chat(self, user_prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content.strip()


def load_chat_model(model_name, backend="hf", base_url=None, temperature=0.0, top_p=1.0,
                    max_new_tokens=256) -> ChatModel:
    if backend == "hf":
        return HFChatModel(model_name, temperature, top_p, max_new_tokens)
    if backend == "openai":
        if not base_url:
            raise ValueError("--base_url is required for the openai backend.")
        return OpenAICompatibleChatModel(model_name, base_url, temperature, top_p, max_new_tokens)
    raise ValueError(f"Unknown backend: {backend}")


def add_backend_args(parser):
    parser.add_argument("--backend", choices=["hf", "openai"], default="hf")
    parser.add_argument("--base_url", default=None, help="Server URL for the openai backend.")
