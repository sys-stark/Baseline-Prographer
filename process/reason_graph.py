import random
import time
from collections import deque
import collections
import numpy as np
import torch
from openai import OpenAI
from torch_geometric.nn import GNNExplainer

from process.match.deephunter.graphmatchnet import GraphMatchingScorer
from process.match.test_model import build_model
from process.match.evaluation import compute_similarity
from process.match.test_model import get_default_config


# TODO 1、把真实数据数据集成上来 2、串联匹配可解释性 3、triple 4、tuoyu测试
def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def find_important_nodes_and_edges(graph_edge_mask, edge_index, top_k=5, verbose=True):
    """
    计算图中节点和边的重要性得分，并返回映射

    Args:
        graph_edge_mask (Tensor): 边的重要性分数 (shape: [num_edges])
        edge_index (Tensor): 边的索引矩阵 (shape: [2, num_edges])
        top_k (int): 打印前 top_k 个重要节点和边
        verbose (bool): 是否打印详细信息

    Returns:
        node_importance_dict: {node_id: importance_score}
        edge_importance_list: [((src, dst), score), ...]
    """
    edge_mask_np = graph_edge_mask.detach().cpu().numpy()
    edge_index_np = edge_index.detach().cpu().numpy()
    # 节点重要性初始化
    num_nodes = edge_index_np.max() + 1
    node_importance = [0.0] * num_nodes

    # 累加边的重要性到相邻节点
    for idx, score in enumerate(edge_mask_np):
        src, dst = edge_index_np[:, idx]
        node_importance[src] += score
        node_importance[dst] += score
    # 构建映射结果
    node_importance_dict = {i: score for i, score in enumerate(node_importance)}
    edge_importance_list = [((int(edge_index_np[0, i]), int(edge_index_np[1, i])), float(score)) for i, score in
                            enumerate(edge_mask_np)]

    # 排序
    sorted_nodes = sorted(node_importance_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_edges = sorted(edge_importance_list, key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"📌 Top-{top_k} 重要节点:")
        for nid, score in sorted_nodes[:top_k]:
            print(f"  节点 {nid} → 重要性: {score:.4f}")

        print(f"\n🔗 Top-{top_k} 重要边:")
        for (src, dst), score in sorted_edges[:top_k]:
            print(f"  边 ({src} → {dst}) → 重要性: {score:.4f}")

    return node_importance_dict, edge_importance_list

def call_llm(template):
    """
    通用大模型调用封装：
    - 输入：template（prompt字符串）
    - 输出：LLM完整返回的文本内容
    """
    client = OpenAI(
        api_key="sk-xhUZwtWJmekrtdX2hLvnC6nnuNSfe6qNIidWbzRIQBoZCEMa",
        base_url="https://api.chatanywhere.tech/v1"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": template}]
    )

    answer = response.choices[0].message.content.strip()
    return answer


def bfs_igraph_multi_start(G, start_vertices):
    """
    支持多个起点的 BFS，记录所有完整路径
    :param graph: igraph.Graph 对象
    :param select_k: 每个节点随机选择的邻居数量
    :return: 所有完整路径 (list)
    """
    paths = {}  # 记录每个节点ID的完整路径
    final_paths = []  # 存放完整路径结果

    # BFS 初始化
    visited = set()
    queue = deque()

    # 初始化多个起点
    print(f"初始节点{start_vertices}")
    for start in start_vertices:
        queue.append(start)
        visited.add(start)
        paths[start] = [start]
    while queue:
        node = queue.popleft()
        neighbors_idx = set()
        for nbr_idx in G.neighbors(int(node), mode="ALL"):
            if nbr_idx not in visited and nbr_idx not in neighbors_idx:
                neighbors_idx.add(nbr_idx)
        print(f"node{node} have neighbors{neighbors_idx}")
        # 探索
        if neighbors_idx:
            # 随机选择 K 个邻居扩展
            # ✅ TODO: LLM 控制选择策略（示例：LLM 让你选或筛选 neighbor）随机选择 K 个邻居扩展
            neighbors_with_relation = []
            for neighbor_idx in neighbors_idx:
                try:
                    edge_id = G.get_eid(node, neighbor_idx, directed=False)
                    relation = G.es[edge_id]["actions"]
                    neighbors_with_relation.append((node, relation, neighbor_idx))
                except Exception as e:
                    print(f" 无法获取边 {node} -> {neighbor_idx}：{e}")
            selected_neighbors = llm_select_neighbors(G, node, neighbors_with_relation, paths[node])
            print(
                f"node {node} 随机选择后三元组 {selected_neighbors}")
            for src, relation, dst in selected_neighbors:
                visited.add(dst)
                queue.append(dst)
                paths[dst] = paths[node] + [relation, dst]  # 累加三元组路径
                print(f"探索的路径为 src: {src} dst: {dst}, path[dst]: {paths[dst]}")

        # 终止判断
        temp_path = "->".join(map(str, paths[node]))
        final_paths = [p for p in final_paths if not temp_path.startswith(p + "->")]
        final_paths.append(temp_path)
        if llm_should_stop(G, final_paths):
            print("LLM判定停止，BFS退出")
            break

    print(f"\n最终完整路径集合: {final_paths}")
    return final_paths


def llm_select_neighbors(G, current_node, candidate_triples, current_path, llm = False):
    """
    调用大模型 LLM 决策：从候选邻居三元组中选择要走的边
    :param current_node: 当前节点
    :param candidate_triples: [(src, relation, dst), ...]
    :param current_path: 当前已走的路径（三元组路径）
    :return: LLM 选择的三元组列表
    """
    if llm:
        # 格式化路径和候选边为字符串
        str_to_id_map = {}
        triples_str_parts = []

        for s, r, o in candidate_triples:
            s_prop = G.vs[s]["properties"]
            o_prop = G.vs[o]["properties"]
            triple_str = f"('{s_prop}', '{r}', '{o_prop}')"
            triples_str_parts.append(triple_str)
            str_to_id_map[(s_prop, r, o_prop)] = (s, r, o)

        triples_str = ", ".join(triples_str_parts)
        template = (
            f"当前节点为：{current_node}\n"
            f"当前已走路径为：{current_path}\n"
            f"候选三元组为：[{triples_str}]\n"
            "请从候选三元组中选择你认为最优的（可选择多个），\n"
            "直接返回 Python 列表格式，例如： [('A', 'rel1', 'B'), ('B', 'rel2', 'C')]。"
        )

        # 调用大模型
        response = call_llm(template)
        print(f"LLM选择邻居回复：{response}")

        # 尝试解析返回值为三元组列表
        try:
            selected = eval(response)
            # 还原字符串三元组 → 原始 ID 三元组
            selected_triples = []
            for triple in selected:
                if isinstance(triple, tuple) and len(triple) == 3:
                    key = tuple(triple)
                    if key in str_to_id_map:
                        selected_triples.append(str_to_id_map[key])
            if selected_triples:
                return selected_triples
        except Exception as e:
            print(f"LLM返回无法解析或匹配失败，默认随机选：{e}")
    else:
        # 如果 LLM 返回出错，随机 fallback
        select_k = 2
        return random.sample(candidate_triples, min(select_k, len(candidate_triples)))

def convert_path_ids_to_names(path_lines, G):
    converted = []
    nodeName = []
    for line in path_lines:
        parts = line.strip().split("->")
        name_parts = []
        try:
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # 偶数位：节点 ID → 名字
                    name_parts.append(G.vs[int(part)]["properties"])
                    nodeName.append(G.vs[int(part)]['name'])
                else:
                    # 奇数位：边名，原样保留
                    name_parts.append(part)
            converted.append("->".join(name_parts))
        except (ValueError, IndexError, KeyError) as e:
            converted.append(f"[无效路径] {line}")
    with open("node_names.txt", "w", encoding="utf-8") as f:
        for name in nodeName:
            f.write(name + "\n")
    return converted

def llm_should_stop(G, final_paths, llm = False):
    """
    调用大模型 LLM 判断：根据当前完整路径集合，决定是否停止BFS
    大模型会基于以下规则作答：
    - 如果路径数量超过3条，建议停止
    - 如果路径中包含关键节点 'J'，建议停止
    """
    # 动态拼接路径列表到 prompt 中
    final_paths_names = convert_path_ids_to_names(final_paths, G)
    if llm:
        template = (
            f"以下是当前完整路径集合：{final_paths_names}。\n"
            "请判断：是否应该停止遍历？\n"
            "规则：如果路径数量超过5条 或 路径中包含关键节点 'J'，或者单条路径长度超过5, 则建议停止。\n"
            "请直接回答：是 或 否。"
        )

        # 调用封装好的 LLM
        response = call_llm(template)
        print(f"final_paths: {final_paths} 大模型回复：{response}")

        # 自动识别LLM回答
        if "是" in response or "yes" in response.lower():
            print("LLM判定：停止")
            return True
        else:
            print("LLM判定：继续搜索")
            return False
    else:
        if len(final_paths) > 5:
            print("路径总数 > 5，停止。")
            return True
        print("硬规则判定：继续搜索")
        return False

def _pack_batch(graphs, node_embeddings, edge_embeddings):
    """Pack a batch of graphs into a single `GraphData` instance.
Args:
  graphs: a list of generated networkx graphs.
Returns:
  graph_data: a `GraphData` instance, with node and edge indices properly
    shifted.
"""
    Graphs = []
    for graph in graphs:
        Graphs.append(graph)
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

def get_graph(batch):
    if len(batch) != 2:
        # if isinstance(batch, GraphData):
        graph = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        return node_features, edge_features, from_idx, to_idx, graph_idx
    else:
        graph, labels = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        labels = torch.from_numpy(labels).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx, labels

def reson_test_model(pair, node_embeddings, edge_embeddings, model_path="saved_model.pth"):
    start = time.time()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    config = get_default_config()
    # 确定 feature 维度
    node_feature_dim = 30
    edge_feature_dim = 30

    model = build_model(config, node_feature_dim, edge_feature_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 单独获取一个 pair 进行预测和解释
    packed_graphs = _pack_batch(pair,node_embeddings, edge_embeddings)
    node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(packed_graphs)
    edge_index = torch.stack([from_idx, to_idx], dim=0).to(device)

    # 模型预测
    with torch.no_grad():
        eval_pairs = model(
            x=node_features.to(device),
            edge_index=edge_index,
            batch=None,
            graph_idx=graph_idx.to(device),
            edge_features=edge_features.to(device),
            n_graphs=2  # 一个 pair，两个图
        )
        x, y = reshape_and_split_tensor(eval_pairs, 2)
        similarity = compute_similarity(config, x, y)
        print(f"图对相似度: {similarity.item():.4f}")

    # 解释模型输出
    explainer = GNNExplainer(model, epochs=200)
    model.eval()
    with torch.backends.cudnn.flags(enabled=False):
        feat_mask, edge_mask = explainer.explain_graph(
            node_features.to(device),
            edge_index,
            graph_idx=graph_idx.to(device),
            edge_features=edge_features.to(device),
            n_graphs=2
        )
    print("图的特征重要性:", feat_mask)
    print("图的边重要性:", edge_mask)

    # 第2张图的节点索引
    second_graph_nodes = (graph_idx == 1).nonzero(as_tuple=True)[0]
    node_id_map = {old.item(): new for new, old in enumerate(second_graph_nodes)}
    edge_mask_idx = []
    for i, (src, dst) in enumerate(edge_index.t()):  # 遍历所有边
        if src.item() in node_id_map and dst.item() in node_id_map:
            edge_mask_idx.append(i)
    edge_mask_idx = torch.tensor(edge_mask_idx, dtype=torch.long)
    second_raw_graph_edge_index = edge_index[:, edge_mask_idx]
    mapped_src = [node_id_map[src.item()] for src in second_raw_graph_edge_index[0]]
    mapped_dst = [node_id_map[dst.item()] for dst in second_raw_graph_edge_index[1]]
    second_graph_edge_index = torch.tensor([mapped_src, mapped_dst], dtype=torch.long)

    second_graph_edge_mask = edge_mask[edge_mask_idx]
    node_scores, edge_scores = find_important_nodes_and_edges(second_graph_edge_mask, second_graph_edge_index)
    # 按重要性得分倒序排列节点
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    # 取前 k 个重要节点
    k = 5  # 可以根据需要设置
    top_k_nodes = [node_id for node_id, _ in sorted_nodes[:k]]
    final_full_paths = bfs_igraph_multi_start(pair[1], top_k_nodes)
    print("\n最终完整路径:")
    for path in final_full_paths:
        print(path)
    print("\n最终完整路径名称:")
    converted = convert_path_ids_to_names(final_full_paths, pair[1])
    for path in converted:
        print(path)


