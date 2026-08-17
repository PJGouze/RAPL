# src/model/text_encoders/biobert_encoder/text_encoder.py

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


class BioBERTEncoder:
    """
    Text encoder based on a BioBERT-compatible Hugging Face model.

    The class exposes the same interface as MiniLML6Encoder so that it
    can be used by the existing RAPL embedding pipeline.
    """

    def __init__(
        self,
        model_name,
        device,
        normalize=True,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        self.embedding_dim = int(
            self.model.config.hidden_size
        )

    @torch.no_grad()
    def embed(self, text_list):
        """
        Encode a list of texts into sentence-level embeddings.

        Parameters
        ----------
        text_list : list[str]
            Texts to encode.

        Returns
        -------
        torch.Tensor
            Shape:
            (len(text_list), embedding_dim)
        """

        if len(text_list) == 0:
            return torch.zeros(
                0,
                self.embedding_dim,
            )

        batch = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**batch)

        token_embeddings = outputs.last_hidden_state

        attention_mask = (
            batch["attention_mask"]
            .unsqueeze(-1)
            .to(token_embeddings.dtype)
        )

        embeddings = (
            token_embeddings * attention_mask
        ).sum(dim=1)

        denominator = attention_mask.sum(
            dim=1
        ).clamp(min=1e-9)

        embeddings = embeddings / denominator

        if self.normalize:
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1,
            )

        return embeddings.cpu()

    def __call__(
        self,
        q_text,
        text_entity_list,
        relation_list,
    ):
        q_emb = self.embed([q_text])

        entity_embs = self.embed(
            text_entity_list
        )

        relation_embs = self.embed(
            relation_list
        )

        return (
            q_emb,
            entity_embs,
            relation_embs,
        )

    @torch.no_grad()
    def encode_questions(
        self,
        questions,
        batch_size=64,
    ):
        """
        Encode questions in batches.

        Returns
        -------
        list[torch.Tensor]
            One tensor of shape
            (1, embedding_dim)
            per question.
        """

        all_embs = []

        for i in range(
            0,
            len(questions),
            batch_size,
        ):
            batch = questions[
                i:i + batch_size
            ]

            embeddings = self.embed(batch)

            all_embs.extend(
                embedding.unsqueeze(0)
                for embedding in embeddings
            )

        return all_embs
