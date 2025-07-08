import abc
import contextlib
import random
import collections
import copy

import numpy as np
import networkx as nx

"""A general Interface"""


class GraphSimilarityDataset(object):
    """Base class for all the graph similarity learning datasets.
  This class defines some common interfaces a graph similarity dataset can have,
  in particular the functions that creates iterators over pairs and triplets.
  """
    @abc.abstractmethod
    def pairs(self, batch_size, G, node_embeddings, edge_embeddings):
        """Create an iterator over pairs.
    Args:
      batch_size: int, number of pairs in a batch.
    Yields:
      graphs: a `GraphData` instance.  The batch of pairs put together.  Each
        pair has 2 graphs (x, y).  The batch contains `batch_size` number of
        pairs, hence `2*batch_size` many graphs.
      labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
    """
        pass


"""Graph Edit Distance Task"""


# Graph Manipulation Functions
def permute_graph_nodes(g):
    """Permute node ordering of a graph, returns a new graph."""
    n = g.number_of_nodes()
    new_g = nx.Graph()
    new_g.add_nodes_from(range(n))
    perm = np.random.permutation(n)
    edges = g.edges()
    new_edges = []
    for x, y in edges:
        new_edges.append((perm[x], perm[y]))
    new_g.add_edges_from(new_edges)
    return new_g


def substitute_random_edges(g, n):
    """Substitutes n edges from graph g with another n randomly picked edges."""
    g = copy.deepcopy(g)
    n_nodes = g.number_of_nodes()
    edges = list(g.edges())
    # sample n edges without replacement
    e_remove = [
        edges[i] for i in np.random.choice(np.arange(len(edges)), n, replace=False)
    ]
    edge_set = set(edges)
    e_add = set()
    while len(e_add) < n:
        e = np.random.choice(n_nodes, 2, replace=False)
        # make sure e does not exist and is not already chosen to be added
        if (
                (e[0], e[1]) not in edge_set
                and (e[1], e[0]) not in edge_set
                and (e[0], e[1]) not in e_add
                and (e[1], e[0]) not in e_add
        ):
            e_add.add((e[0], e[1]))

    for i, j in e_remove:
        g.remove_edge(i, j)
    for i, j in e_add:
        g.add_edge(i, j)
    return g


def substitute_random_edges_ig(G, G_add, positive, ratio=0.1):
    """在 igraph.Graph (有向图) 中随机替换 `n` 条边"""
    G = G.copy()  # 复制图，避免修改原始图
    n_nodes = G.vcount()  # 获取节点数
    edges = G.get_edgelist()  # 获取所有边的列表

    # # 如果图的节点数小于 2，无法替换边，直接返回原图
    # if n_nodes < 2:
    #     print(f"[警告] 图的节点数不足 ({n_nodes})，无法替换边，直接返回原图。")
    #     return G
    #
    # # 如果要替换的边比图中的边还多，直接返回
    # if len(edges) < n:
    #     print(f"[警告] 需要替换 {n} 条边，但图中只有 {len(edges)} 条边，直接返回原图。")
    #     return G

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
    # 加点
    if not positive:
        # 获取 G_add 中的节点数与 G 中的原节点数
        n_nodes_add = G_add.vcount()
        n_nodes_origin = G.vcount()
        nodes_to_add_count = min(n_nodes_add, nodes_to_remove_count)  # 要添加的节点数量与删除的节点数量相同
        # 遍历添加节点到 G 中
        new_nodes = []
        for node_add in range(nodes_to_add_count):
            G.add_vertex()  # 添加一个新节点
            new_node_index = len(G.vs) - 1  # 获取新节点的索引
            # 复制 G_add 中对应节点的属性到新节点
            G.vs[new_node_index]['name'] = G_add.vs[node_add]['name']
            G.vs[new_node_index]['type'] = G_add.vs[node_add]['type']
            G.vs[new_node_index]['properties'] = G_add.vs[node_add]['properties']
            new_nodes.append(new_node_index)  # 保存新节点的索引
        # 将新添加的节点与现有节点连接
        new_edges = []
        for node in new_nodes:
            # 随机选择一个已存在的节点来连接
            existing_node = np.random.choice(n_nodes_origin)  # 从原有节点中随机选择一个节点
            new_edges.append((existing_node, node))
        G.add_edges(new_edges)

    return G  # 返回修改后的图


class GraphEditDistanceDataset(GraphSimilarityDataset):
    """Graph edit distance dataset."""

    def __init__(
            self,
            n_nodes_range,
            p_edge_range,
            n_changes_positive,
            n_changes_negative,
            communities,
            permute=True,
    ):
        """Constructor.
    Args:
      n_nodes_range: a tuple (n_min, n_max).  The minimum and maximum number of
        nodes in a graph to generate.
      p_edge_range: a tuple (p_min, p_max).  The minimum and maximum edge
        probability.
      n_changes_positive: the number of edge substitutions for a pair to be
        considered positive (similar).
      n_changes_negative: the number of edge substitutions for a pair to be
        considered negative (not similar).
      permute: if True (default), permute node orderings in addition to
        changing edges; if False, the node orderings across a pair or triplet of
        graphs will be the same, useful for visualization.
    """
        self._n_min, self._n_max = n_nodes_range
        self._p_min, self._p_max = p_edge_range
        self._k_pos = n_changes_positive
        self._k_neg = n_changes_negative
        self._communities = communities
        self._permute = permute

    def _get_pair(self, positive, communities_list, idx, G):
        """Generate one pair of graphs from a given community structure.

        Args:
            positive (bool): 是否是正样本
            communities (dict): {社区ID: [节点列表]}
            G (igraph.Graph): 原始的完整图

        Returns:
            permuted_g (igraph.Graph): 经过节点重排的社区子图
            changed_g (igraph.Graph): 经过边修改的社区子图
        """
        # 从 `communities` 里选一个社区
        if idx > len(communities_list)-1 :
            idx = 0  # 可选：循环利用
        community_nodes = communities_list[idx]
        g = G.subgraph(community_nodes)

        if idx +1  > len(communities_list)-1 :
            next = 0
        else:
            next = idx + 1
        changed_community_nodes = communities_list[next]
        g_add = G.subgraph(changed_community_nodes)

        # 根据 `positive` 选择边的修改数量
        n_changes = self._k_pos if positive else self._k_neg

        # 对子图 `g` 进行边修改
        changed_g = substitute_random_edges_ig(g, g_add, positive, n_changes)

        return g, changed_g

    def _pairs(self, batch_size, G, node_embeddings, edge_embeddings):
        """Yields batches of pair data."""
        community_list = list(self._communities.values())  # 顺序列表
        idx = 0

        while True:
            batch_graphs = []
            batch_labels = []
            positive = True
            for _ in range(batch_size):
                if idx + 1 >= len(community_list):
                    idx = 0  # 重头开始

                g1, g2 = self._get_pair(positive, community_list, idx, G)
                batch_graphs.append((g1, g2))
                batch_labels.append(1 if positive else -1)
                positive = not positive

            packed_graphs = self._pack_batch(batch_graphs, node_embeddings, edge_embeddings)
            if packed_graphs is None:
                # 如果这次采样失败，继续下一轮
                print("[警告] 本次采样生成了空batch，跳过...")
                continue
            labels = np.array(batch_labels, dtype=np.int32)
            yield packed_graphs, labels

    def _pack_batch(self, graphs, node_embeddings, edge_embeddings):
        """Pack a batch of graphs into a single `GraphData` instance.
    Args:
      graphs: a list of generated networkx graphs.
    Returns:
      graph_data: a `GraphData` instance, with node and edge indices properly
        shifted.
    """
        Graphs = []
        for graph in graphs:
            for inergraph in graph:
                Graphs.append(inergraph)
        graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []
        node_names = []
        edge_relations = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.vcount()
            n_edges = g.ecount()
            # 检查是否为空图
            if n_nodes == 0:
                print(f"[警告] 图 {i} 没有节点，跳过！")
                continue

            if n_edges == 0:
                print(f"[警告] 图 {i} 没有边，跳过！")
                continue

            edges = np.array(g.get_edgelist(), dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
            node_names.extend([g.vs[j]['name'] for j in range(n_nodes)])
            edge_relations.extend([
                g.es[k]['actions'] if 'actions' in g.es[k].attributes() else "undefined_relation"
                for k in range(n_edges)
            ])

            n_total_nodes += n_nodes
            n_total_edges += n_edges


        GraphData = collections.namedtuple('GraphData', [
            'from_idx',
            'to_idx',
            'node_features',
            'edge_features',
            'graph_idx',
            'n_graphs'])

        if not from_idx:
            return None

        node_feature_list = []
        for name in node_names:
            if name in node_embeddings:
                node_feature_list.append(node_embeddings[name])
            else:
                # 填一个正确维度的全1向量（假设维度是 30）
                node_feature_list.append(np.ones(30, dtype=np.float32))

        node_features = np.array(node_feature_list, dtype=np.float32)
        edge_feature_list = []
        for name in edge_relations:
            if name in edge_embeddings:
                edge_feature_list.append(edge_embeddings[name])
            else:
                # 填一个正确维度的全1向量
                edge_feature_list.append(np.ones(30, dtype=np.float32))
        edge_features = np.array(edge_feature_list, dtype=np.float32)

        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features.
            # setting higher dimension of ones to confirm code functioning
            # with high dimensional features.
            # node_features=np.ones((n_total_nodes, 8), dtype=np.float32),
            node_features=node_features,
            # edge_features=np.ones((n_total_edges, 4), dtype=np.float32),
            edge_features=edge_features,
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )


# Use Fixed datasets for evaluation
@contextlib.contextmanager
def reset_random_state(seed):
    """This function creates a context that uses the given seed."""
    np_rnd_state = np.random.get_state()
    rnd_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed + 1)
    try:
        yield
    finally:
        random.setstate(rnd_state)
        np.random.set_state(np_rnd_state)


class FixedGraphEditDistanceDataset(GraphEditDistanceDataset):
    """A fixed dataset of pairs or triplets for the graph edit distance task.
  This dataset can be used for evaluation.
  """

    def __init__(
            self,
            n_nodes_range,
            p_edge_range,
            n_changes_positive,
            n_changes_negative,
            dataset_size,
            communities,
            permute=True,
            seed=1234,
    ):
        super(FixedGraphEditDistanceDataset, self).__init__(
            n_nodes_range,
            p_edge_range,
            n_changes_positive,
            n_changes_negative,
            communities,
            permute=permute,
        )
        self._dataset_size = dataset_size
        self._seed = seed

    def pairs(self, batch_size, G, node_embeddings, edge_embeddings):
        """Yield pairs and labels."""
        if hasattr(self, "_pairs") and hasattr(self, "_labels"):
            pairs = self._pairs
            labels = self._labels
        else:
            # get a fixed set of pairs first
            with reset_random_state(self._seed):
                community_list = list(self._communities.values())
                pairs = []
                labels = []
                positive = True
                idx = 0
                for _ in range(self._dataset_size):
                    pairs.append(self._get_pair(positive, community_list, idx, G))
                    idx += 1
                    if idx == len(community_list):
                        idx = 0
                    labels.append(1 if positive else -1)
                    positive = not positive
            labels = np.array(labels, dtype=np.int32)
            self._pairs = pairs
            self._labels = labels

        ptr = 0
        while ptr + batch_size <= len(pairs):
            batch_graphs = pairs[ptr: ptr + batch_size]
            packed_batch = self._pack_batch(batch_graphs, node_embeddings, edge_embeddings)
            if packed_batch is None:
                # 如果这次采样失败，继续下一轮
                print("[警告] 本次采样生成了空batch，跳过...")
                ptr += batch_size
                continue
            yield packed_batch, labels[ptr: ptr + batch_size]
            ptr += batch_size

        # # 补上最后不足一个 batch 的部分
        if ptr < len(pairs):
            batch_graphs = pairs[ptr:]
            packed_batch = self._pack_batch(batch_graphs, node_embeddings, edge_embeddings)
            if packed_batch is not None:
                yield packed_batch, labels[ptr:]
