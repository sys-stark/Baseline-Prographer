import igraph as ig
import leidenalg as la
from process.type_enum import ObjectType

resource_types = {ObjectType.NETFLOW_OBJECT.value, ObjectType.FILE_OBJECT_BLOCK.value, ObjectType.MemoryObject.value}

def create_process_graph():
    """创建一个进程间关系的有向图"""
    G = ig.Graph(directed=True)

    # 添加进程节点
    # processes = [
    #     {"name": "Process0", "type": "process"},
    #     {"name": "Process1", "type": "process"},
    #     {"name": "Process2", "type": "process"},
    #     {"name": "Process3", "type": "process"},
    #     {"name": "Process4", "type": "process"},
    #     {"name": "FileA", "type": "resource"},  # 文件资源节点
    #     {"name": "Process5", "type": "process"},
    #     {"name": "Process6", "type": "process"},
    #     {"name": "Process7", "type": "process"},
    # ]
    processes = [
        {"name": "Process0", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Process1", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Process2", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Process3", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Process4", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Socket", "type": ObjectType.NETFLOW_OBJECT.value},  # 套接字资源
        {"name": "File", "type": ObjectType.FILE_OBJECT_BLOCK.value},  # 文件资源
    ]

    G.add_vertices(len(processes))
    for i, process in enumerate(processes):
        G.vs[i]["name"] = process["name"]
        G.vs[i]["type"] = process["type"]  # 区分进程与资源

    # edges = [
    #     ("Process0", "Process1"),  # 基本父子
    #     ("Process1", "Process2"),  # 基本父子
    #     ("Process1", "Process3"),  # 基本父子
    #     ("Process2", "Process4"),  # 资源依赖
    #     ("Process0", "FileA"),  # 进程2 访问了 文件A（资源依赖）
    #     ("Process4", "FileA"),  # 进程4 访问了 文件A（资源依赖）
    #     ("Process3", "Process5"),  # 基本父子
    #     ("Process3", "Process6"),  # 基本父子
    #     ("Process3", "Process7"),  # 基本父子
    # ]
    edges = [
        ("Process0", "Socket"),  # Process0 连接到 Socket
        ("Process1", "Socket"),  # Process1 通过 Socket 通信
        ("Process4", "Process1"),  # Process1 连接 Process4
        ("Process4", "Process2"),  # Process2 连接 Process4
        ("File", "Process3"),  # Process3 访问 File
        ("File", "Process2"),  # File 影响 Process2
        ("Process0", "File"),  # 进程0 访问了文件（资源依赖）
        ("File", "Process1"),  # 进程1 访问了文件（资源依赖）
    ]
    # 添加边（基本父子关系 + 资源依赖）
    G.add_edges(edges)

    return G


def set_weight(G):
    # 设置默认权重
    set_default_weight(G)
    # 设置进程-进程权重
    set_process_weights(G)
    # # 设置进程-资源权重
    set_resource_weights(G)


def is_resource_dependent(G, source, target):
    """
    判断 source 和 target 是否通过资源形成无向环，并返回环的大小。
    - 如果 source 到 target 存在路径，并且路径上包含 resource 节点，则判定为资源依赖。
    - 返回 (是否存在资源依赖, 资源依赖环的最小大小)。
    """
    source_idx = G.vs.find(name=source).index
    target_idx = G.vs.find(name=target).index

    # 转换为无向图，确保搜索时不受边方向影响
    G_undirected = G.as_undirected()
    # 获取所有路径（在无向图中）
    paths = get_all_paths(G_undirected, source_idx, target_idx)

    min_loop_size = float("inf")  # 记录最小环大小
    has_resource_dependency = False  # 是否有资源依赖

    for path in paths:
        # 检查路径中是否有资源节点
        if any((G_undirected.vs[node_idx]["type"] in resource_types) for node_idx in path):
            has_resource_dependency = True  # 发现路径中有资源
            min_loop_size = min(min_loop_size, len(path))  # 记录最短的环大小

    if has_resource_dependency:
        return True, min_loop_size  # 返回是否存在资源依赖 & 环大小
    else:
        return False, -1  # 没有资源依赖，环大小返回 -1


def set_resource_weights(G, W_base=1.0):
    """
    统计每个资源节点被哪些进程访问，并根据类别的进程数量调整边权重。
    - 进程类别是根据无向图的连通性确定的。
    - 直接根据类别的进程数量计算权重，无需额外遍历每个边两次。
    """
    resource_access = {}  # { 资源节点 -> 访问它的进程列表 }

    # **遍历所有节点，找到资源节点**
    for v in G.vs:
        if v["type"] in resource_types:
            resource_access[v["name"]] = []  # 初始化该资源的访问进程列表

    # **遍历所有边，找到进程 -> 资源的访问关系**
    for edge in G.es:
        source = G.vs[edge.source]["name"]
        target = G.vs[edge.target]["name"]

        if G.vs[edge.source]["type"] == ObjectType.SUBJECT_PROCESS.value and G.vs[edge.target]["type"] in resource_types:
            resource_access[target].append(source)  # 资源被 source 进程访问
        elif G.vs[edge.source]["type"] in resource_types and G.vs[edge.target]["type"] == ObjectType.SUBJECT_PROCESS.value:
            resource_access[source].append(target)  # 资源被 target 进程访问

    # **创建无向图**
    G_undirected = G.as_undirected()

    # **为每个资源节点，分类访问该资源的进程**
    for resource, accessing_processes in resource_access.items():
        process_clusters = classify_processes_by_common_ancestor(G, accessing_processes)

        # **计算类别大小**
        cluster_sizes = {frozenset(cluster): len(cluster) for cluster in process_clusters}

        # **创建一个字典存储每个进程所属类别的权重**
        process_weights = {
            proc: W_base * cluster_sizes[frozenset(cluster)]
            for cluster in process_clusters for proc in cluster
        }

        total_weight = sum(process_weights.values())

        if total_weight > 0:
            process_weights = {proc: weight / total_weight for proc, weight in process_weights.items()}

        # **直接遍历 resource 的边，并分配权重**
        for edge in G.es:
            source = G.vs[edge.source]["name"]
            target = G.vs[edge.target]["name"]

            # **如果是进程 -> 资源 或 资源 -> 进程**
            if (source == resource and target in accessing_processes) or (
                    target == resource and source in accessing_processes):
                edge["weight"] = process_weights[target] if target in process_weights else process_weights[source]


def set_process_weights(G, W_base=1.0, delta_factor=5):
    """
    计算进程间的权重（基于进程节点维度），并按比例分配到边：
    - 先计算每个进程的 总权重
    - 然后 按比例归一化，确保每条边的权重反映进程的重要性。
    """
    process_weights = {}  # 存储每个进程对其他进程的初始权重
    total_weights = {}  # 存储每个进程的总权重（用于归一化）

    # **遍历所有进程节点**
    for source in G.vs:
        if source["type"] != ObjectType.SUBJECT_PROCESS.value:
            continue  # 只处理进程节点

        source_name = source["name"]
        process_weights[source_name] = {}  # 初始化该进程的权重字典
        total_weights[source_name] = 0  # 初始化总权重

        for target_idx in G.neighbors(source, mode="out"):
            target = G.vs[target_idx]
            if target["type"] != ObjectType.SUBJECT_PROCESS.value or source_name == target["name"]:
                continue  # 只处理进程-进程关系，跳过自己

            target_name = target["name"]

            # 判断是否存在资源依赖
            resDepFlag, distance = is_resource_dependent(G, source_name, target_name)

            if resDepFlag:
                delta = delta_factor/ distance  # 计算 Δ
                weight = W_base * (1 + delta)  # **资源依赖增益**
            else:
                weight = W_base  # **基本父子关系**

            # **存储进程间的权重**
            process_weights[source_name][target_name] = weight
            total_weights[source_name] += weight  # 计算总权重

    # **归一化，并将权重赋值到边**
    for edge in G.es:
        source_name = G.vs[edge.source]["name"]
        target_name = G.vs[edge.target]["name"]

        if source_name in process_weights and target_name in process_weights[source_name]:
            if total_weights[source_name] > 0:  # 避免除以 0
                edge["weight"] = process_weights[source_name][target_name] / total_weights[source_name]
            else:
                edge["weight"] = 0  # 如果总权重为 0，则权重设为 0


def get_connected_processes(G_undirected, start_proc, all_procs):
    """
    获取与 start_proc 互相可达的所有进程（形成一个类别）。

    参数:
    - G_undirected: 无向图
    - start_proc: 需要查找的起始进程
    - all_procs: 资源连接的所有进程列表

    返回:
    - set(可达的进程集合)
    """
    try:
        start_idx = G_undirected.vs.find(name=start_proc).index
        reachable_idxs = G_undirected.subcomponent(start_idx)  # 获取无向连通子图
        reachable_procs = {G_undirected.vs[idx]["name"] for idx in reachable_idxs if
                           G_undirected.vs[idx]["name"] in all_procs}
        return reachable_procs
    except ValueError:
        return set()  # 进程不存在，返回空集合


def is_related(G, proc1, proc2):
    """
    判断两个进程是否有血缘关系（无向路径）：
    - 如果 proc1 到 proc2 之间存在路径（无向路径），则认为它们是亲属进程。
    - 否则，认为它们是无关进程。
    """
    try:
        # 获取进程索引
        source_idx = G.vs.find(name=proc1).index
        target_idx = G.vs.find(name=proc2).index

        # **转换为无向图**，忽略边的方向
        G_undirected = G.as_undirected()

        # 获取所有无向最短路径
        paths = G_undirected.get_all_shortest_paths(source_idx, to=target_idx)

        return len(paths) > 0  # 只要存在路径，说明有血缘关系
    except ValueError:
        return False  # 进程不存在


def print_communities(communities):
    """打印社区划分结果"""
    for cid, nodes in communities.items():
        print(f"Community {cid}: {nodes}")


def detect_communities(G):
    set_weight(G)

    """使用 Modularity Method 执行 Leiden 社区检测"""
    partition = la.find_partition(G, la.ModularityVertexPartition, weights='weight')

    # 解析社区划分
    communities = {i: [] for i in set(partition.membership)}
    for node, community_id in enumerate(partition.membership):
        communities[community_id].append(G.vs[node]["name"])

    print_communities(communities)
    return communities

def detect_communities_with_id(G):
    set_weight(G)
    """使用 Modularity Method 执行 Leiden 社区检测"""
    partition = la.find_partition(G, la.ModularityVertexPartition, weights='weight')
    # 解析社区划分
    communities = {i: [] for i in set(partition.membership)}
    for node, community_id in enumerate(partition.membership):
        communities[community_id].append(G.vs[node])
    print_communities(communities)
    return communities


def set_default_weight(G, weight=1.0):
    """
    设置图中所有边的权重为指定值（默认 1.0）。
    """
    G.es["weight"] = [weight] * len(G.es)


def print_graph_info(G):
    """打印图的结构和边的权重"""
    print("Edges with weights:")
    for edge in G.es:
        source = G.vs[edge.source]["name"]
        target = G.vs[edge.target]["name"]
        weight = edge["weight"]
        print(f"{source} -> {target}, Weight: {weight:.4f}")


def get_all_paths(G, source_idx, target_idx, path=None, visited=None, max_depth=10, max_steps=1000, step_counter=[0]):
    """
    使用递归方式查找所有从 source_idx 到 target_idx 的路径，并限制最大递归深度和总探索次数。

    参数：
    - G: igraph 图对象
    - source_idx: 起始节点索引
    - target_idx: 目标节点索引
    - path: 当前路径（递归内部使用）
    - visited: 当前访问的节点集合（防止环）
    - max_depth: 最大路径深度限制
    - max_steps: 最大递归尝试次数（防止爆栈）
    - step_counter: 用于计数递归次数的列表（避免不可变类型）

    返回：
    - 所有合法路径的列表，每个路径是一个节点索引列表
    """
    if path is None:
        path = []
    if visited is None:
        visited = set()

    # 超过探索次数，提前终止
    if step_counter[0] >= max_steps:
        return []

    step_counter[0] += 1

    if source_idx in visited:
        return []

    if len(path) > max_depth:
        return []

    path.append(source_idx)
    visited.add(source_idx)

    all_paths = []
    if source_idx == target_idx:
        all_paths.append(path[:])
    else:
        for neighbor in G.neighbors(source_idx, mode="all"):
            all_paths.extend(
                get_all_paths(G, neighbor, target_idx, path, visited, max_depth, max_steps, step_counter)
            )

    path.pop()
    visited.remove(source_idx)

    return all_paths


def find_ancestors(G, proc):
    """
    查找进程的所有祖先（所有向上的父进程）。

    参数:
    - G: igraph 有向图
    - proc: 进程名称

    返回:
    - ancestors: 祖先进程集合
    """
    try:
        proc_idx = G.vs.find(name=proc).index
    except ValueError:
        return set()  # 进程不存在，返回空集

    ancestors = set()
    queue = [proc_idx]

    while queue:
        current = queue.pop(0)
        for parent in G.predecessors(current):  # 获取所有父进程
            parent_name = G.vs[parent]["name"]
            if parent_name not in ancestors:
                ancestors.add(parent_name)
                queue.append(parent)  # 继续向上查找

    return ancestors

def classify_processes_by_common_ancestor(G, accessing_processes):
    """
    根据共同祖先进程，对访问相同资源的进程进行分类。

    参数:
    - G: igraph 有向图
    - accessing_processes: 访问某个资源的进程集合

    返回:
    - process_clusters: 进程分类的列表，每个类别是一个集合
    """
    process_clusters = []  # 存储多个类别的进程
    visited = set()

    # **存储每个进程的祖先集合**
    ancestor_map = {proc: find_ancestors(G, proc) for proc in accessing_processes}

    # **遍历 accessing_processes**
    for proc in accessing_processes:
        if proc in visited:
            continue  # 跳过已分类的进程

        # **获取当前进程的祖先**
        proc_ancestors = ancestor_map[proc]

        # **找到所有具有相同祖先的进程**
        cluster = {p for p in accessing_processes if not proc_ancestors.isdisjoint(ancestor_map[p])}
        cluster.add(proc)
        # **存储分类结果**
        process_clusters.append(cluster)
        visited.update(cluster)  # 标记已分类的进程

    return process_clusters

# test
# G = create_process_graph()
# set_weight(G)
# print_graph_info(G)
# communities = detect_communities(G)
