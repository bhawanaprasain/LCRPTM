import torch.nn as nn


def find_token_embedding_module(model):
    """
    Find the token embedding module in a Hugging Face / custom causal LM.

    Returns:
        embedding_path: dot path to the embedding module
        embedding_module: nn.Embedding instance
    """
    candidate_paths = [
        "transformer.wte",
        "model.transformer.wte",
        "wte",
        "model.wte",
        "embed_tokens",
        "model.embed_tokens",
        "tok_embeddings",
        "model.tok_embeddings",
    ]

    for path in candidate_paths:
        obj = model
        found = True

        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break

        if found and isinstance(obj, nn.Embedding):
            return path, obj

    return None, None


def get_module_by_path(model, module_path):
    """
    Resolve a nested module from a dot-separated path.
    """
    obj = model
    for part in module_path.split("."):
        obj = getattr(obj, part)
    return obj


def set_module_by_path(model, module_path, new_module):
    """
    Replace a nested module using a dot-separated path.
    """
    obj = model
    parts = module_path.split(".")

    for part in parts[:-1]:
        obj = getattr(obj, part)

    setattr(obj, parts[-1], new_module)


def make_get_input_embeddings(embedding_path):
    """
    Create a get_input_embeddings method bound to a discovered embedding path.
    """

    def get_input_embeddings(self):
        return get_module_by_path(self, embedding_path)

    return get_input_embeddings


def make_set_input_embeddings(embedding_path):
    """
    Create a set_input_embeddings method bound to a discovered embedding path.
    """

    def set_input_embeddings(self, new_embedding):
        set_module_by_path(self, embedding_path, new_embedding)

    return set_input_embeddings


from types import MethodType


def patch_model_for_resize(model):
    embedding_path, embedding_module = find_token_embedding_module(model)

    if embedding_module is None:
        raise RuntimeError("Could not find token embedding module in model.")

    model.get_input_embeddings = MethodType(
        make_get_input_embeddings(embedding_path), model
    )
    model.set_input_embeddings = MethodType(
        make_set_input_embeddings(embedding_path), model
    )

    if hasattr(model, "lm_head"):

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, new_head):
            self.lm_head = new_head

        model.get_output_embeddings = MethodType(get_output_embeddings, model)
        model.set_output_embeddings = MethodType(set_output_embeddings, model)

    return model
