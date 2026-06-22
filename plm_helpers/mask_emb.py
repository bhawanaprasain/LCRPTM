class OldTokenGradMask:
    """
    Zero out gradients for original vocabulary tokens so that
    only newly added token embeddings are updated.
    """

    def __init__(self, old_vocab_size: int):
        self.old_vocab_size = old_vocab_size

    def __call__(self, grad):
        grad = grad.clone()  # avoid modifying gradient in-place
        grad[: self.old_vocab_size] = 0
        return grad


def register_old_embedding_grad_mask(embedding_layer, n_new_tokens: int):
    """
    Register gradient mask hook on embedding layer so only new tokens train.
    """
    vocab_size = embedding_layer.weight.shape[0]
    old_vocab_size = vocab_size - n_new_tokens

    embedding_layer.weight.register_hook(OldTokenGradMask(old_vocab_size))
