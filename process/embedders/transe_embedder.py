import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from .base import GraphEmbedderBase
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
        triples.append((head, relation, tail))

    return np.array(triples, dtype=object)

def train_embedding_model(triples):
    """
    训练 TransE 知识图谱嵌入模型
    :param triples: list of (head, relation, tail) 三元组
    :return: 训练好的模型, triples_factory
    """
    # 创建 TriplesFactory
    triples_factory = TriplesFactory.from_labeled_triples(triples)
    training, validation, testing = triples_factory.split([0.8, 0.1, 0.1])
    # 训练模型
    result = pipeline(
        model='TransE',
        training=triples_factory,
        validation=validation,
        testing=testing,
        model_kwargs={'embedding_dim': 30},
        training_kwargs={'num_epochs': 100, 'batch_size': 16},
    )

    return result.model, triples_factory

def get_feature_vector(model, triples_factory, name):
    """
    获取特定实体或关系的嵌入向量
    :param model: 训练好的模型
    :param triples_factory: 训练数据的 TriplesFactory
    :param name: 实体或关系名称
    :return: 嵌入向量
    """
    # 检查实体
    if name in triples_factory.entity_to_id:
        entity_id = triples_factory.entity_to_id[name]
        entity_tensor = torch.tensor([entity_id], dtype=torch.long)
        embedding = model.entity_representations[0](
            indices=entity_tensor
        ).detach().cpu().numpy()
        return embedding[0]  # 返回一维数组

    # 检查关系*
    elif name in triples_factory.relation_to_id:
        relation_id = triples_factory.relation_to_id[name]
        relation_tensor = torch.tensor([relation_id], dtype=torch.long)
        embedding = model.relation_representations[0](
            indices=relation_tensor
        ).detach().cpu().numpy()
        return embedding[0]  # 返回一维数组

    else:
        raise ValueError(f"'{name}' 既不是实体也不是关系，请检查输入！")

class TransEEmbedder(GraphEmbedderBase):
    def __init__(self, G, features, mapp):
        super().__init__(G, features, mapp)
        self.model = None
        self.factory = None

    def train(self):
        triples = graph_to_triples(self.G, self.features, self.mapp)
        self.model, self.factory = train_embedding_model(triples)

    def embed_nodes(self):
        node_embeddings = {}
        node_name_list = [v["name"] for v in self.G.vs]
        node_feature_list = [str(self.features[self.mapp.index(name)]) for name in node_name_list]
        for node_name, node_feature in zip(node_name_list, node_feature_list):
            embedding = get_feature_vector(self.model, self.factory, node_feature)
            node_embeddings[node_name] = embedding  # 使用node_name作为键存储embedding
            print(f"Node '{node_name}' embedding: {embedding[:5]}")  # 只显示前5维的embedding
        return node_embeddings

    def embed_edges(self):
        edge_embeddings = {}
        edge_list = [edge['actions'] if 'actions' in edge.attributes() else "undefined_relation" for edge in self.G.es]
        for relation in edge_list:
            embedding = get_feature_vector(self.model, self.factory, relation)
            edge_embeddings[relation] = embedding
            print(f"Relation '{relation}' embedding: {embedding[:5]}")  # 打印前5维
        return edge_embeddings




