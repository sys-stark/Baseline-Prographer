from gensim.models import Word2Vec
from .base import GraphEmbedderBase
import numpy as np

import numpy as np

def graph_to_triples(G, features, mapp):
    """
    将 iGraph 图转换为 (头实体, 关系, 尾实体) 三元组
    :param G: ig.Graph 实例
    :return: list of triples (head, relation, tail)
    """
    triples = []

    for edge in G.es:
        head_id = edge.source  # 获取起点 ID
        tail_id = edge.target  # 获取终点 ID
        relation = edge['actions'] if 'actions' in edge.attributes() else "undefined_relation"  # 关系属性

        # 获取实体的 name（如果有）
        head = str(features[mapp.index(G.vs[head_id]['name'])])
        tail = str(features[mapp.index(G.vs[tail_id]['name'])])
        triples.append([head, relation, tail])

    return triples

class Word2VecEmbedder(GraphEmbedderBase):
    def __init__(self, G, features, mapp):
        super().__init__(G, features, mapp)
        self.model = None

    def train(self):
        phrases = graph_to_triples(self.G, self.features, self.mapp)
        self.model = Word2Vec(sentences=phrases, vector_size=30, window=5, min_count=1, workers=4, epochs=100)

    def embed_nodes(self):
        node_embeddings = {}
        for v in self.G.vs:
            name = v["name"]
            try:
                phrase = self.features[self.mapp.index(name)]
                emb = self.model.wv.infer_vector(phrase)
            except Exception:
                emb = np.zeros(30)
            node_embeddings[name] = emb
        return node_embeddings

    def embed_edges(self):
        edge_embeddings = {}
        for edge in self.G.es:
            relation = edge['actions'] if 'actions' in edge.attributes() else "undefined_relation"
            if relation in self.model.wv:
                embedding = self.model.wv[relation]
            else:
                embedding = np.zeros(30)
            edge_embeddings[relation] = embedding
        return edge_embeddings