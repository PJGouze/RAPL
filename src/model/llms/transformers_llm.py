#update: 06/08 10:05

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
        print(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if model_name == 'Qwen/Qwen2.5-0.5B-Instruct':

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32
            ).to(self.device)

        elif model_name == 'Qwen/Qwen3.5-35B-A3B-FP8':

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype="auto",
                trust_remote_code=True
            )
        else:

            self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        dtype="auto",
                        trust_remote_code=True
                    )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()


    @torch.no_grad()
    def generate(
    self,
    messages,
    max_new_tokens=256
):
        """
        Generate responses from one or multiple conversations.

        Parameters
        ----------
        messages :
            Either:
            - list[dict] for one conversation
            - list[list[dict]] for a batch

        Returns
        -------
        str or list[str]
        """

        # Detect single example
        single = False

        if isinstance(messages[0], dict):
            messages = [messages]
            single = True


        # Build prompts
        prompts = [
            self.tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=True
            )
            for m in messages
        ]



        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)


        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # print("="*80)
        # print("Prompt length:", inputs["input_ids"].shape[1])
        # print("Output length:", outputs.shape[1])

        # generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

        # print("Generated tokens:", generated_tokens.shape)

        # print("Raw output:")
        # print(self.tokenizer.decode(outputs[0], skip_special_tokens=False))

        # print("="*80)


        answers = []

        for output, input_ids in zip(
            outputs,
            inputs["input_ids"]
        ):


            generated_tokens = output[input_ids.shape[-1]:]

            answer = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            answers.append(answer.strip())


        if single:
            return answers[0]

        return answers
