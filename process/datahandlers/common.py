import os
from collections import defaultdict


def merge_properties(src_dict, target_dict):
    for k, v in src_dict.items():
        if k not in target_dict:
            target_dict[k] = v

def collect_dot_paths(base_dir):
    result = []
    for file in os.listdir(base_dir):   #遍历
        full_path = os.path.join(base_dir, file)
        if os.path.isfile(full_path) and file.endswith(".dot"):  # 如果文件是 .dot 格式
            result.append(full_path)  # 使用 append 添加文件路径到列表
    return result  # 返回 .dot 文件路径的列表

def collect_json_paths(base_dir):
    result = {}
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            result[subdir] = {"benign": [], "malicious": []}
            for category in ["benign", "malicious"]:
                category_path = os.path.join(subdir_path, category)
                if os.path.exists(category_path):
                    for file in os.listdir(category_path):
                        if file.endswith(".json") and not file.startswith("._"):
                            full_path = os.path.join(category_path, file)
                            result[subdir][category].append(full_path)
    return result

def collect_atlas_label_paths(base_dir):
    result = dict()
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".txt"):
                    full_path = os.path.join(subdir_path, file)
                    name_only = os.path.splitext(file)[0]
                    result[name_only] = full_path
    return result

def collect_label_paths(base_dir):
    result = dict()
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            category_path = os.path.join(subdir_path, "malicious")
            if os.path.exists(category_path):
                for file in os.listdir(category_path):
                    if file.endswith(".txt") and not file.startswith("._"):
                        full_path = os.path.join(category_path, file)
                        result[subdir] = full_path
    return result


# =================处理特征成图=========================
def extract_properties(node_id, row, action, netobj2pro, subject2pro, file2pro):
    if node_id in netobj2pro:
        return netobj2pro[node_id]
    elif node_id in file2pro:
        return file2pro[node_id]
    elif node_id in subject2pro:
        return subject2pro[node_id]
    else:
        return " ".join(
            [row.get('exec', ''), action] + ([row.get('path')] if row.get('path') else [])
        )
        # return [row.get('exec', ''), action] + ([row.get('path')] if row.get('path') else [])

def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)


def get_or_add_node(G, node_id, node_type, properties):
    """
    查找图中是否已有节点 node_id：
    - 如果有，返回该节点索引，并更新属性
    - 如果没有，添加新节点并返回其索引
    """
    try:
        v = G.vs.find(name=node_id)
        v['properties'] = properties  # 可选更新属性
        return v.index
    except ValueError:
        G.add_vertex(name=node_id, type=node_type, properties=properties)
        return len(G.vs) - 1

def add_edge_if_new(G, src, dst, action):
    """
    向图 G 添加一条从 src 到 dst 的边，附带 action 属性。
    - 若边已存在且包含该 action，不做任何处理。
    - 若边已存在但未包含该 action 再添加一条边
    - 若边不存在，则添加边并设置 action。
    """
    if G.are_connected(src, dst):
        eids = G.get_eids([(src, dst)], directed=True, error=False)
        for eid in eids:
            if G.es[eid]["actions"] == action:
                return  # 该 action 已存在，不重复添加
    G.add_edge(src, dst)
    G.es[-1]["actions"] = action

def update_edge_index(edges, edge_index, index, relations, relations_index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

        relation = relations[(src_id, dst_id)]
        relations_index[(src, dst)] = relation