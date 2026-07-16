import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from .base import BaseLLM


class TransformersLLM(BaseLLM):
    """
    Wrapper around Hugging Face causal language models.

    This class is designed to load instruction-tuned models such as
    Qwen/Qwen2.5-0.5B-Instruct and generate responses from chat messages.
    """

    def __init__(
        self,
        model_name,
        device="cpu"
    ):

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        ).to(self.device)

        self.model.eval()


    @torch.no_grad()
    def generate(
        self,
        messages,
        max_new_tokens=256
    ):
        """
        Generate a response from chat messages.

        Parameters
        ----------
        messages : list[dict]
            Conversation history.

            Example:
            [
                {
                    "role": "user",
                    "content": "What is diabetes?"
                }
            ]

        max_new_tokens : int
            Maximum number of tokens generated.

        Returns
        -------
        str
            Generated answer.
        """

        # Convert messages into Qwen chat format
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)


        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )


        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]


        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )


        return answer.strip()