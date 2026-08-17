#update: 03/08 10:05
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


class MiniLML6Encoder:
    """
    Text encoder based on sentence-transformers/all-MiniLM-L6-v2.

    This class provides the same interface as GTELargeEN so that it can be
    used interchangeably within the RAPL embedding pipeline.
    """

    def __init__(self, model_name, device, normalize=True):
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.model.eval()

        self.normalize = normalize

    @torch.no_grad()
    def embed(self, text_list):
        """
        Encode a list of texts into sentence embeddings.

        Parameters
        ----------
        text_list : list[str]
            List of texts to encode.

        Returns
        -------
        torch.Tensor
            Tensor of shape (len(text_list), 384).
        """

        if len(text_list) == 0:
            return torch.zeros(0, 384)

        batch = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**batch)

        token_embeddings = outputs.last_hidden_state

        attention_mask = batch["attention_mask"].unsqueeze(-1)

        # Mean pooling
        emb = (
            token_embeddings * attention_mask
        ).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu()

    def __call__(self, q_text, text_entity_list, relation_list):
        """
        Encode a question, entity list, and relation list.

        Parameters
        ----------
        q_text : str
            Question text.

        text_entity_list : list[str]
            List of textual entity names.

        relation_list : list[str]
            List of textual relation names.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - q_emb: shape (1, 384)
            - entity_embs: shape (n_entities, 384)
            - relation_embs: shape (n_relations, 384)
        """

        q_emb = self.embed([q_text])
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)

        return q_emb, entity_embs, relation_embs

    @torch.no_grad()
    def encode_questions(self, questions, batch_size=64):
        """
        Encode a list of questions in batches.

        Returns
        -------
        list[Tensor]
            One embedding tensor (1,384) per question.
        """
        all_embs = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            emb = self.embed(batch)
            all_embs.extend([e.unsqueeze(0) for e in emb])

        return all_embs
