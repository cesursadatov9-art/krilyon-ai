import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    MODEL_ID,
    MAX_CONTEXT_LENGTH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
)


class NEXModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.float16 if self.device == "cuda"
            else torch.float32
        )

        print("KrilyonNEX | NEX-1")
        print(f"Device: {self.device}")
        print(f"Model yüklənir: {MODEL_ID}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()

        print("Model hazırdır!")

    @torch.inference_mode()
    def chat(self, message, history=None):
        messages = []

        for user_message, assistant_message in (history or []):
            messages.append({
                "role": "user",
                "content": user_message
            })
            messages.append({
                "role": "assistant",
                "content": assistant_message
            })

        messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=self.tokenizer.eos_token_id
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()


if __name__ == "__main__":
    ai = NEXModel()

    while True:
        try:
            message = input("\nSən: ").strip()

            if message.lower() in {"çıx", "exit", "quit"}:
                break

            if message:
                print("NEX-1:", ai.chat(message))

        except (KeyboardInterrupt, EOFError):
            print("\nSöhbət bitdi.")
            break
