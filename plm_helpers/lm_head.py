import torch.nn as nn

import torch


class SplitTokenEmbedding(nn.Module):
    def __init__(self, base_emb: nn.Embedding, old_vocab: int, n_new: int):
        super().__init__()
        self.old_vocab = old_vocab
        self.hidden = base_emb.embedding_dim

        # frozen old part
        self.old_emb = nn.Embedding(old_vocab, self.hidden)
        self.old_emb.weight.data.copy_(base_emb.weight.data[:old_vocab])
        self.old_emb.weight.requires_grad_(False)

        # trainable new part
        self.new_emb = nn.Embedding(n_new, self.hidden)
        self.new_emb.weight.data.copy_(
            base_emb.weight.data[old_vocab : old_vocab + n_new]
        )
        self.new_emb.weight.requires_grad_(True)

    def forward(self, input_ids: torch.Tensor):
        out = torch.zeros(
            (*input_ids.shape, self.hidden),
            device=input_ids.device,
            dtype=self.new_emb.weight.dtype,
        )

        old_mask = input_ids < self.old_vocab
        new_mask = ~old_mask

        if old_mask.any():
            out[old_mask] = self.old_emb(input_ids[old_mask])

        if new_mask.any():
            new_ids = input_ids[new_mask] - self.old_vocab
            out[new_mask] = self.new_emb(new_ids)

        return out


class SplitLMHead(nn.Module):
    """
    Frozen old-vocab linear + trainable new-vocab linear.
    Produces logits for full vocab_size = old_vocab + n_new.
    """

    def __init__(self, base_head: nn.Module, hidden: int, old_vocab: int, n_new: int):
        super().__init__()
        self.old_vocab = old_vocab
        self.n_new = n_new
        self.hidden = hidden

        # base_head is usually nn.Linear(hidden, vocab, bias=False)
        W = base_head.weight.data  # (vocab, hidden)

        self.old_head = nn.Linear(hidden, old_vocab, bias=False)
        self.old_head.weight.data.copy_(W[:old_vocab])
        self.old_head.weight.requires_grad_(False)

        self.new_head = nn.Linear(hidden, n_new, bias=False)
        self.new_head.weight.data.copy_(W[old_vocab : old_vocab + n_new])
        self.new_head.weight.requires_grad_(True)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: (..., hidden)
        old_logits = self.old_head(hidden_states)  # (..., old_vocab)
        new_logits = self.new_head(hidden_states)  # (..., n_new)
        return torch.cat([old_logits, new_logits], dim=-1)
