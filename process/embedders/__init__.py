from .word2vec_embedder import Word2VecEmbedder
from .transe_embedder import TransEEmbedder
from .prographer_embedder import ProGrapherEmbedder

def get_embedder_by_name(name: str):
    name = name.lower()
    if name == "word2vec":
        return Word2VecEmbedder
    elif name == "transe":
        return TransEEmbedder
    elif name == "prographer":
        return ProGrapherEmbedder
    else:
        raise ValueError(f"未知编码器类型: {name}")