import numpy as np
from process.datahandlers import get_handler
from process.embedders import get_embedder_by_name
from process.partition import detect_communities_with_id
from reason_graph import reson_test_model
# TODO 1、把真实数据数据集成上来 2、串联匹配可解释性 3、triple 4、tuoyu测试

def substitute_random_edges_ig(G, ratio=0.1):
    """在 igraph.Graph (有向图) 中随机替换 `n` 条边"""
    G = G.copy()  # 复制图，避免修改原始图
    n_nodes = G.vcount()  # 获取节点数
    edges = G.get_edgelist()  # 获取所有边的列表

    ############################操作边#################################################
    # 1、随机选择 `n` 条边进行删除
    total_edges = len(edges)
    n_changes_edges= int(total_edges * ratio)
    # 1、随机选择 `n_changes_edges` 条边进行删除
    e_remove_idx = np.random.choice(total_edges, n_changes_edges, replace=False)  # 选 `n_changes_edges` 条边索引
    e_remove = [edges[i] for i in e_remove_idx]  # 获取要删除的边
    edge_set = set(map(tuple, edges))  # 转换为集合，方便查重

    # 2、随机生成 `n_changes_edges` 条新边，确保新边不重复且不和删除的边相同
    e_add = set()
    while len(e_add) < n_changes_edges:
        e = tuple(np.random.choice(n_nodes, 2, replace=False))  # 生成 (src, dst)
        if e not in edge_set and e not in e_remove and e not in e_add:  # 确保新边不重复且与删除的边不同
            e_add.add(e)

    # 3、执行删除和添加
    G.delete_edges(e_remove)  # 删除选定的 `n` 条边
    G.add_edges(list(e_add))  # 添加 `n` 条新边

    #############################操作点##################################
    # 删点 关联的边删掉
    # 删除节点的比例
    nodes_to_remove_count = int(n_nodes * ratio)  # 按比例确定删除的节点数
    nodes_to_remove = np.random.choice(G.vs.indices, size=nodes_to_remove_count, replace=False)  # 随机选取节点
    nodes_to_remove_set = set(nodes_to_remove)
    G.delete_vertices(nodes_to_remove_set)
    return G  # 返回修改后的图

def get_pair(G, ratio = 0.1):
    """Generate one pair of graphs from a given community structure.

    Args:
        positive (bool): 是否是正样本
        communities (dict): {社区ID: [节点列表]}
        G (igraph.Graph): 原始的完整图

    Returns:
        permuted_g (igraph.Graph): 经过节点重排的社区子图
        changed_g (igraph.Graph): 经过边修改的社区子图
    """
    # 对子图 `g` 进行边修改
    """在 igraph.Graph (有向图) 中随机替换 `n` 条边"""
    G_c = G.copy()  # 复制图，避免修改原始图
    n_nodes = G_c.vcount()  # 获取节点数
    edges = G_c.get_edgelist()  # 获取所有边的列表

    ############################操作边#################################################
    # 1、随机选择 `n` 条边进行删除
    total_edges = len(edges)
    n_changes_edges = int(total_edges * ratio)
    # 1、随机选择 `n_changes_edges` 条边进行删除
    e_remove_idx = np.random.choice(total_edges, n_changes_edges, replace=False)  # 选 `n_changes_edges` 条边索引
    e_remove = [edges[i] for i in e_remove_idx]  # 获取要删除的边
    edge_set = set(map(tuple, edges))  # 转换为集合，方便查重

    # 2、随机生成 `n_changes_edges` 条新边，确保新边不重复且不和删除的边相同
    e_add = set()
    while len(e_add) < n_changes_edges:
        e = tuple(np.random.choice(n_nodes, 2, replace=False))  # 生成 (src, dst)
        if e not in edge_set and e not in e_remove and e not in e_add:  # 确保新边不重复且与删除的边不同
            e_add.add(e)

    # 3、执行删除和添加
    G_c.delete_edges(e_remove)  # 删除选定的 `n` 条边
    G_c.add_edges(list(e_add))  # 添加 `n` 条新边

    #############################操作点##################################
    # 删点 关联的边删掉
    # 删除节点的比例
    nodes_to_remove_count = int(n_nodes * ratio)  # 按比例确定删除的节点数
    nodes_to_remove = np.random.choice(G_c.vs.indices, size=nodes_to_remove_count, replace=False)  # 随机选取节点
    nodes_to_remove_set = set(nodes_to_remove)
    G_c.delete_vertices(nodes_to_remove_set)

    return G, G_c

def construct_test_graph_pair(G):
    communities = detect_communities_with_id(G)
    malicious_communities = []
    benign_communities = []
    for community_id in communities:
        members = communities[community_id]
        subgraph = G.subgraph(members)
        labels = []
        for v in subgraph.vs:
            label = v["label"] if "label" in v.attributes() else None
            labels.append(label)
        # TODO
        if any(lbl == 1 for lbl in labels):
            malicious_communities.append(subgraph)
        else:
            benign_communities.append(subgraph)

    if not malicious_communities:
        raise ValueError("未找到包含恶意节点的社区。")

    # 具有恶意label的图保留label在删减下其他边 得到查询图 和原图当作被查询图
    pair_list = []
    for malicious_community in malicious_communities:
        query_graph, provence_graph = get_pair(malicious_community)
        pair_list.append((query_graph, provence_graph))
    return pair_list

# 获取数据集
data_handler = get_handler("atlas", False)
# data_handler = get_handler("theia", False)
# 加载数据
data_handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G = data_handler.build_graph()
# 中选择恶意图对配合测试
pair_list = construct_test_graph_pair(G)

# 嵌入构造特征向量
embedder_class = get_embedder_by_name("word2vec")
# embedder_class = get_embedder_by_name("transe")
embedder = embedder_class(G, features, mapp)
embedder.train()
node_embeddings = embedder.embed_nodes()
edge_embeddings = embedder.embed_edges()

for pair in pair_list:
    reson_test_model(pair, node_embeddings, edge_embeddings)
