# D:\baseline\process\datahandlers\atlas_handler.py

import os.path
import pandas as pd
import igraph as ig
import re

from .base import BaseProcessor
from .common import merge_properties, collect_dot_paths, extract_properties, collect_atlas_label_paths
from .type_enum import ObjectType

# 假设 collect_nodes_from_log 和 collect_edges_from_log 在这个文件中
# 如果它们在别处，你需要调整导入路径
def collect_nodes_from_log(paths):
    netobj2pro, subject2pro, file2pro = {}, {}, {}
    domain_name_set, ip_set, connection_set, session_set, web_object_set = {}, {}, {}, {}, {}
    nodes = []
    with open(paths, 'r', encoding='utf-8') as f:
        content = f.read()
    statements = content.split(';')
    node_pattern = re.compile(r'^\s*"?(.+?)"?\s*\[.*?type="?([^",\]]+)"?', re.IGNORECASE)
    for stmt in statements:
        if 'capacity=' in stmt: continue
        match = node_pattern.search(stmt)
        if match:
            node_name, node_typen = match.group(1), match.group(2)
            nodes.append((node_name, node_typen))
    for node_name, node_typen in nodes:
        node_id, node_type = node_name, node_typen
        if node_type == 'domain_name': netobj2pro[node_id] = node_id; domain_name_set[node_id] = node_id
        elif node_type == 'IP_Address': netobj2pro[node_id] = node_id; ip_set[node_id] = node_id
        elif node_type == 'connection': netobj2pro[node_id] = node_id; connection_set[node_id] = node_id
        elif node_type == 'session': netobj2pro[node_id] = node_id; session_set[node_id] = node_id
        elif node_type == 'web_object': netobj2pro[node_id] = node_id; web_object_set[node_id] = node_id
        elif node_type == 'process': subject2pro[node_id] = node_id
        elif node_type == 'file': file2pro[node_id] = node_id
    return netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set

def collect_edges_from_log(paths, domain_name_set, ip_set, connection_set, session_set, web_object_set, subject2pro, file2pro) -> pd.DataFrame:
    edges = []
    with open(paths, "r", encoding="utf-8") as f:
        content = f.read()
    statements = content.split(";")
    edge_pattern = re.compile(r'"?([^"]+)"?\s*->\s*"?(.*?)"?\s*\[.*?capacity=.*?type="?([^",\]]+)"?.*?timestamp=(\d+)', re.IGNORECASE | re.DOTALL)
    for stmt in statements:
        if "capacity=" not in stmt: continue
        m = edge_pattern.search(stmt)
        if m:
            source, target, edge_type, ts = (x.strip() for x in m.groups())
            source_type = "PRINCIPAL_LOCAL"
            if source in domain_name_set or source in ip_set or source in connection_set or source in session_set or source in web_object_set: source_type = "NETFLOW_OBJECT"
            elif source in subject2pro: source_type = "SUBJECT_PROCESS"
            elif source in file2pro: source_type = "FILE_OBJECT_BLOCK"
            target_type = "PRINCIPAL_LOCAL"
            if target in domain_name_set or target in ip_set or target in connection_set or target in session_set or target in web_object_set: target_type = "NETFLOW_OBJECT"
            elif target in subject2pro: target_type = "SUBJECT_PROCESS"
            elif target in file2pro: target_type = "FILE_OBJECT_BLOCK"
            edges.append((source, source_type, target, target_type, edge_type, int(ts)))
    return pd.DataFrame(edges, columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"])


class ATLASHandler(BaseProcessor):
    def __init__(self, base_path=None, train=True):
        """【修改】初始化用于图级别追踪的变量"""
        super().__init__(base_path, train)
        # 用于存储每个图(dot文件)的DataFrame
        self.all_dfs_map = {}
        # 用于存储图(dot文件)被处理的顺序
        self.graph_names_in_order = []
        # 用于存储每个生成的快照属于哪个图
        self.snapshot_to_graph_map = []
        # 其他实例变量保持不变
        self.all_labels = []
        self.all_netobj2pro = {}
        self.all_subject2pro = {}
        self.all_file2pro = {}
        self.total_loaded_bytes = 0

    def load(self):
        """【重构】此方法现在只加载数据，不再合并，为逐图处理做准备。"""
        print("处理 ATLAS 数据集...")
        graph_files = collect_dot_paths(self.base_path) #获取所有 .dot 数据文件的路径列表
        label_map = collect_atlas_label_paths(self.base_path) #获取标签文件的路径映射字典

        # 清空之前的数据
        self.all_dfs_map.clear()
        self.graph_names_in_order.clear()
        self.all_labels.clear()

        for dot_file in graph_files:
            # 你可以保留或删除这里的测试代码
            # if "M1-CVE-2015-5122_windows_h1" not in dot_file:
            #     continue

            self.total_loaded_bytes += os.path.getsize(dot_file)
            dot_name = os.path.splitext(os.path.basename(dot_file))[0]
            print(f"正在加载场景: {dot_name}")

            self.graph_names_in_order.append(dot_name) #将这个图的名称添加到 self.graph_names_in_order 列表中，记录下处理的顺序。

            if not self.train:
                if dot_name in label_map:
                    with open(label_map[dot_name], 'r', encoding='utf-8') as label_file:
                        self.all_labels.extend([line.strip() for line in label_file if line.strip()])  #读取其中的每一行（恶意实体名），并将其添加到全局的 self.all_labels 列表中。
                else:
                    print(f"  - 警告: 未找到场景 '{dot_name}' 的标签文件。")
          #从当前的 .dot 文件中解析出所有的节点和边信息。
            netobj2pro, subject2pro, file2pro, dns, ips, conns, sess, webs = collect_nodes_from_log(dot_file)
            df = collect_edges_from_log(dot_file, dns, ips, conns, sess, webs, subject2pro, file2pro)

            self.all_dfs_map[dot_name] = df

            merge_properties(netobj2pro, self.all_netobj2pro)
            merge_properties(subject2pro, self.all_subject2pro)
            merge_properties(file2pro, self.all_file2pro)

        print(f"所有 {len(self.graph_names_in_order)} 个图的数据加载完毕。")

    def build_graph(self):
        """【重构】按顺序逐个图地生成快照，并记录快照与图的映射关系。"""
        # --- 全局返回值初始化 ---
        self.snapshots = []   #用来存放最终所有生成的 igraph.Graph 快照对象。
        self.snapshot_to_graph_map = []   #用来存放每个快照的“主人”是谁（即图的名称）。

        # --- 按加载顺序，逐个图处理 ---
        for graph_name in self.graph_names_in_order:
            print(f"\n--- 正在为图 '{graph_name}' 构建快照 ---")
            df = self.all_dfs_map.get(graph_name)
            if df is None or df.empty:
                print("  - 该图无数据，跳过。")
                continue

            # --- 为每个图重置快照生成器状态 ---
            snapshot_size = 300
            forgetting_rate = 0.3
            self.cache_graph = ig.Graph(directed=True)
            self.node_timestamps = {}
            self.first_flag = True

            sorted_df = df.sort_values(by='timestamp') if 'timestamp' in df.columns else df  #将当前图的所有事件（DataFrame df）按照 timestamp 列进行排序
            # --- 主事件循环 (只针对当前图的df) ---
            for _, row in sorted_df.iterrows():  #遍历排序后的每一个事件（每一行）
                # 从当前行 row 中提取出关键信息
                actor_id, object_id = row["actorID"], row["objectID"]
                action, timestamp = row["action"], row.get('timestamp', 0)

                # --- 添加节点到缓存图 (逻辑与原来相同) ---
                try: self.cache_graph.vs.find(name=actor_id)  #查看节点是否存在,如果存在直接跳到152行
                except ValueError:
                    actor_type_enum = ObjectType[row['actor_type']] #获取节点的类型
                    self.cache_graph.add_vertex(name=actor_id, type=actor_type_enum.value, type_name=actor_type_enum.name, properties=extract_properties(actor_id, row, action, self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)) # 在图中添加一个新顶点
                self.node_timestamps[actor_id] = timestamp  #更新时间戳

                try: self.cache_graph.vs.find(name=object_id)  #逻辑同上
                except ValueError:
                    object_type_enum = ObjectType[row['object']]
                    self.cache_graph.add_vertex(name=object_id, type=object_type_enum.value, type_name=object_type_enum.name, properties=extract_properties(object_id, row, action, self.all_netobj2pro, self.all_subject2pro, self.all_file2pro))
                self.node_timestamps[object_id] = timestamp

                # --- 添加边到缓存图 (逻辑与原来相同) ---
                actor_idx, object_idx = self.cache_graph.vs.find(name=actor_id).index, self.cache_graph.vs.find(name=object_id).index # 获取 actor 和 object 节点在图中的整数索引（index）
                if not self.cache_graph.are_connected(actor_idx, object_idx): #检查这两个节点之间是否已经存在一条边
                    self.cache_graph.add_edge(actor_idx, object_idx, actions=action, timestamp=timestamp)  #加边

                # --- 快照生成逻辑 ---
                n_nodes = len(self.cache_graph.vs)
                if self.first_flag and n_nodes >= snapshot_size:
                    self._generate_snapshot(graph_name)
                    self.first_flag = False
                elif not self.first_flag and n_nodes >= snapshot_size * (1 + forgetting_rate):
                    self._retire_old_nodes(snapshot_size, forgetting_rate)
                    self._generate_snapshot(graph_name)

            # 处理该图末尾剩余的节点
            if len(self.cache_graph.vs) > 0:
                self._generate_snapshot(graph_name)

        # --- 后期处理：为所有快照的所有节点打上最终标签 ---
        print("\n正在为所有节点打上最终标签...")
        for snapshot in self.snapshots:
            for v in snapshot.vs:
                v["label"] = int(v["name"] in self.all_labels)
                if 'type_name' not in v.attributes():
                    try: v['type_name'] = ObjectType(v['type']).name
                    except ValueError: v['type_name'] = "UNKNOWN_TYPE"

        print("图构建和打标流程全部完成。")
        # 【修改】返回评估脚本需要的所有信息
        return [], {}, {}, {}, self.snapshots, self.snapshot_to_graph_map, self.graph_names_in_order

    def _retire_old_nodes(self, snapshot_size: int, forgetting_rate: float) -> None:
        """这个函数保持不变"""
        n_nodes_to_remove = int(snapshot_size * forgetting_rate)
        if n_nodes_to_remove <= 0: return
        sorted_nodes = sorted(self.node_timestamps.items(), key=lambda item: item[1])
        nodes_to_remove = [node_id for node_id, _ in sorted_nodes[:n_nodes_to_remove]]
        try:
            indices_to_remove = [self.cache_graph.vs.find(name=name).index for name in nodes_to_remove]
            self.cache_graph.delete_vertices(indices_to_remove)
        except ValueError:
            pass # 节点可能已被删除
        for node_id in nodes_to_remove:
            if node_id in self.node_timestamps:
                del self.node_timestamps[node_id]

    def _generate_snapshot(self, graph_name: str) -> None:
        """【修改】记录快照所属的图"""
        snapshot = self.cache_graph.copy()
        self.snapshots.append(snapshot)
        self.snapshot_to_graph_map.append(graph_name)
def collect_nodes_from_log(paths):  # dot文件的路径
    # 创建字典
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    domain_name_set = {}
    ip_set = {}
    connection_set = {}
    session_set = {}
    web_object_set = {}
    nodes = []

    # 读取整个文件
    with open(paths, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分号分隔，处理每个段落
    statements = content.split(';')

    # 正则表达式匹配节点定义
    node_pattern = re.compile(r'^\s*"?(.+?)"?\s*\[.*?type="?([^",\]]+)"?', re.IGNORECASE)

    for stmt in statements:
        if 'capacity=' in stmt:
            continue  # 跳过包含 capacity 字段的段落
        match = node_pattern.search(stmt)
        if match:
            node_name = match.group(1)
            node_typen = match.group(2)
            nodes.append((node_name, node_typen))
    for node_name, node_typen in nodes:  # 遍历所有的节点
        node_id = node_name  # 节点id赋值
        node_type = node_typen  # 赋值type属性
        # -- 网络流节点 --
        if node_type == 'domain_name':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            domain_name_set[node_id] = nodeproperty
        if node_type == 'IP_Address':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            ip_set[node_id] = nodeproperty
        if node_type == 'connection':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            connection_set[node_id] = nodeproperty
        if node_type == 'session':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            session_set[node_id] = nodeproperty
        if node_type == 'web_object':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            web_object_set[node_id] = nodeproperty
        # -- 进程节点 --
        elif node_type == 'process':
            nodeproperty = node_id
            subject2pro[node_id] = nodeproperty
        # -- 文件节点 --
        elif node_type == 'file':
            nodeproperty = node_id
            file2pro[node_id] = nodeproperty

    return netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set


def collect_edges_from_log(paths, domain_name_set, ip_set, connection_set, session_set, web_object_set, subject2pro,
                           file2pro) -> pd.DataFrame:
    """
    从 DOT-like 日志文件中提取含 capacity 的边，并识别 source/target 属于哪个节点集合。
    返回一个包含 source、target、type、timestamp、source_type、target_type 的 DataFrame。
    """
    # 预定义的节点集合

    edges = []

    with open(paths, "r", encoding="utf-8") as f:
        content = f.read()

    statements = content.split(";")

    edge_pattern = re.compile(
        r'"?([^"]+)"?\s*->\s*"?(.*?)"?\s*\['
        r'.*?capacity=.*?'
        r'type="?([^",\]]+)"?.*?'
        r'timestamp=(\d+)',
        re.IGNORECASE | re.DOTALL
    )

    for stmt in statements:
        if "capacity=" not in stmt:
            continue
        m = edge_pattern.search(stmt)
        if m:
            source, target, edge_type, ts = (x.strip() for x in m.groups())

            # 判断 source/target 所属集合
            if source in domain_name_set:
                source_type = "NETFLOW_OBJECT"
            elif source in ip_set:
                source_type = "NETFLOW_OBJECT"
            elif source in connection_set:
                source_type = "NETFLOW_OBJECT"
            elif source in session_set:
                source_type = "NETFLOW_OBJECT"
            elif source in web_object_set:
                source_type = "NetFlowObject"
            elif source in subject2pro:
                source_type = "SUBJECT_PROCESS"
            elif source in file2pro:
                source_type = "FILE_OBJECT_BLOCK"
            else:
                source_type = "PRINCIPAL_LOCAL"

            if target in domain_name_set:
                target_type = "NETFLOW_OBJECT"
            elif target in ip_set:
                target_type = "NETFLOW_OBJECT"
            elif target in connection_set:
                target_type = "NETFLOW_OBJECT"
            elif target in session_set:
                target_type = "NETFLOW_OBJECT"
            elif target in web_object_set:
                target_type = "NetFlowObject"
            elif target in subject2pro:
                target_type = "SUBJECT_PROCESS"
            elif target in file2pro:
                target_type = "FILE_OBJECT_BLOCK"
            else:
                target_type = "PRINCIPAL_LOCAL"

            edges.append((source, source_type, target, target_type, edge_type, int(ts)))

    return pd.DataFrame(edges, columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"])
