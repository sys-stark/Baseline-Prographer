from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch
import numpy as np
import igraph as ig


# # 定义自定义三元组数据
# triples = np.array([
#     ("ProcessA", "creates", "File1"),
#     ("ProcessA", "communicates_with", "ProcessB"),
#     ("ProcessB", "writes", "File2"),
#     ("ProcessC", "reads", "File1"),
#     ("ProcessC", "creates", "File3"),
#     ("ProcessA", "creates", "File3"),
#     ("ProcessB", "communicates_with", "ProcessC"),
#     ("ProcessD", "creates", "File4"),
#     ("ProcessE", "writes", "File5"),
#     ("ProcessF", "reads", "File2"),
#     ("ProcessG", "creates", "File6"),
#     ("ProcessH", "writes", "File1"),
#     ("ProcessI", "communicates_with", "ProcessJ"),
#     ("ProcessJ", "creates", "File7"),
#     ("ProcessA", "writes", "File8"),
#     ("ProcessD", "reads", "File4"),
#     ("ProcessE", "creates", "File5"),
#     ("ProcessG", "writes", "File9"),
#     ("ProcessH", "reads", "File10"),
#     ("ProcessI", "creates", "File11"),
# ], dtype=object)
#
#
# # 利用三元组数据构造 TriplesFactory
# triples_factory = TriplesFactory.from_labeled_triples(triples)
# training, validation, testing = triples_factory.split([0.8, 0.1, 0.1])
#
# # 运行 pipeline，使用 TransE 模型进行训练
# result = pipeline(
#     model='TransE',
#     training=triples_factory,
#     validation=validation,
#     testing=testing,
#     model_kwargs={'embedding_dim': 5},
#     training_kwargs={'num_epochs': 100, 'batch_size': 16},
# )
#
# trained_model = result.model
# # **获取的实体嵌入**
# entity_to_id = triples_factory.entity_to_id  # 获取实体索引
# entity_id = entity_to_id["ProcessA"]
# entity_tensor = torch.tensor([entity_id], dtype=torch.long)
# entity_embedding = trained_model.entity_representations[0](
#     indices=entity_tensor
# ).detach().cpu().numpy()
# print(f"ProcessA entity embedding: {entity_embedding}")
#
# # ** 获取关系的嵌入**
# relation_to_id = triples_factory.relation_to_id  # 获取关系索引
# relation_id = relation_to_id["creates"]  # 关系的索引
# relation_tensor = torch.tensor([relation_id], dtype=torch.long)  # 转成 tensor
# relation_embedding = trained_model.relation_representations[0](
#     indices=relation_tensor
# ).detach().cpu().numpy()
# print(f"Relation 'creates' embedding: {relation_embedding}")

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

# def custom_split(triples_factory, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
#     # 计算每个子集的大小
#     total_triples = len(triples_factory)
#     train_size = int(total_triples * train_ratio)
#     valid_size = int(total_triples * valid_ratio)
#     test_size = total_triples - train_size - valid_size
#
#     # 获取所有的 triples
#     all_triples = list(triples_factory)
#
#     # 统计所有的实体和关系
#     all_entities = set()
#     all_relations = set()
#     for triple in all_triples:
#         subject, predicate, object_ = triple
#         all_entities.add(subject)
#         all_entities.add(object_)
#         all_relations.add(predicate)
#
#     # 用来追踪训练集、验证集和测试集的实体和关系
#     entities_in_train = set()
#     relations_in_train = set()
#
#     # 初始化训练集、验证集和测试集
#     train_triples = []
#     valid_triples = []
#     test_triples = []
#
#     # 先将训练集填满，确保包含所有实体和关系
#     for triple in all_triples:
#         subject, predicate, object_ = triple
#
#         # 确保每个实体和关系都在训练集中
#         if subject not in entities_in_train or object_ not in entities_in_train or predicate not in relations_in_train:
#             train_triples.append(triple)
#             entities_in_train.add(subject)
#             entities_in_train.add(object_)
#             relations_in_train.add(predicate)
#
#         # 如果训练集已经包含了所有实体和关系，停止填充训练集
#         if len(train_triples) >= train_size and len(entities_in_train) == len(all_entities) and len(
#                 relations_in_train) == len(all_relations):
#             break
#
#     # 确保训练集填满
#     remaining_triples = [triple for triple in all_triples if triple not in train_triples]
#
#     # 分配剩余的 triples 给验证集和测试集
#     for triple in remaining_triples:
#         if len(valid_triples) < valid_size:
#             valid_triples.append(triple)
#         elif len(test_triples) < test_size:
#             test_triples.append(triple)
#
#     # 返回手动分配的训练集、验证集和测试集
#     return TriplesFactory.from_labeled_triples(train_triples), \
#         TriplesFactory.from_labeled_triples(valid_triples), \
#         TriplesFactory.from_labeled_triples(test_triples)

# 训练知识图谱嵌入模型
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


# 2、获取实体/关系的特征向量
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


if __name__ == "__main__":
    # **定义知识图谱数据**
    # triples = np.array([
    #     ("ProcessA", "creates", "File1"),
    #     ("ProcessA", "communicates_with", "ProcessB"),
    #     ("ProcessB", "writes", "File2"),
    #     ("ProcessC", "reads", "File1"),
    #     ("ProcessC", "creates", "File3"),
    #     ("ProcessA", "creates", "File3"),
    #     ("ProcessB", "communicates_with", "ProcessC"),
    #     ("ProcessD", "creates", "File4"),
    #     ("ProcessE", "writes", "File5"),
    #     ("ProcessF", "reads", "File2"),
    #     ("ProcessG", "creates", "File6"),
    #     ("ProcessH", "writes", "File1"),
    #     ("ProcessI", "communicates_with", "ProcessJ"),
    #     ("ProcessJ", "creates", "File7"),
    #     ("ProcessA", "writes", "File8"),
    #     ("ProcessD", "reads", "File4"),
    #     ("ProcessE", "creates", "File5"),
    #     ("ProcessG", "writes", "File9"),
    #     ("ProcessH", "reads", "File10"),
    #     ("ProcessI", "creates", "File11"),
    # ], dtype=object)

    # **1
    G = ig.Graph(directed=True)
    # **添加 10 个节点**
    G.add_vertices(10)
    G.vs[0]['name'] = "ProcessA"
    G.vs[1]['name'] = "ProcessB"
    G.vs[2]['name'] = "ProcessC"
    G.vs[3]['name'] = "ProcessD"
    G.vs[4]['name'] = "ProcessE"
    G.vs[5]['name'] = "File1"
    G.vs[6]['name'] = "File2"
    G.vs[7]['name'] = "File3"
    G.vs[8]['name'] = "File4"
    G.vs[9]['name'] = "File5"

    # **添加边（关系）**
    edges = [
        (0, 5),  # ProcessA -> File1
        (0, 1),  # ProcessA -> ProcessB
        (1, 6),  # ProcessB -> File2
        (1, 2),  # ProcessB -> ProcessC
        (2, 7),  # ProcessC -> File3
        (3, 5),  # ProcessD -> File1
        (3, 8),  # ProcessD -> File4
        (4, 9),  # ProcessE -> File5
        (2, 4),  # ProcessC -> ProcessE
        (0, 3),  # ProcessA -> ProcessD
    ]
    G.add_edges(edges)

    # **为每条边添加动作（关系类型）**
    relations = [
        "creates", "communicates_with", "writes", "communicates_with",
        "reads", "creates", "creates", "writes", "communicates_with", "monitors"
    ]
    G.es["actions"] = relations


    triples = graph_to_triples(G)
    # **训练模型**
    trained_model, triples_factory = train_embedding_model(triples)
    # **获取实体嵌入**
    entities = ["ProcessA", "File1", "ProcessB"]
    for entity in entities:
        embedding = get_feature_vector(trained_model, triples_factory, entity)
        print(f"Entity '{entity}' embedding: {embedding[:5]}")  # 只显示前 5 维

    # **获取关系嵌入**
    relations = ["creates", "communicates_with"]
    for relation in relations:
        embedding = get_feature_vector(trained_model, triples_factory, relation)
        print(f"Relation '{relation}' embedding: {embedding[:5]}")  # 只显示前 5 维
