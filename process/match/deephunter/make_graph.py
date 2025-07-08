import pandas as pd
import json
import igraph as ig
from process.type_enum import ObjectType
import re


# =================处理特征成图=========================
def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)


def update_edge_index(edges, edge_index, index, relations, relations_index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

        relation = relations[(src_id, dst_id)]
        relations_index[(src, dst)] = relation

def extract_properties(node_id, row, action, netobj2pro, subject2pro, file2pro):
    if node_id in netobj2pro:
        return netobj2pro[node_id]
    elif node_id in file2pro:
        return file2pro[node_id]
    elif node_id in subject2pro:
        return subject2pro[node_id]
    else:
        return [row.get('exec', ''), action] + ([row.get('path')] if row.get('path') else [])



# 成图+捕捉特征语料+简化策略这里添加
def prepare_graph_new(df, all_netobj2pro, all_subject2pro, all_file2pro):
    G = ig.Graph(directed=True)
    nodes, edges, relations = {}, [], {}

    for _, row in df.iterrows():
        action = row["action"]

        actor_id = row["actorID"]
        properties = extract_properties(actor_id, row, row["action"], all_netobj2pro, all_subject2pro, all_file2pro)
        add_node_properties(nodes, actor_id, properties)

        object_id = row["objectID"]
        properties1 = extract_properties(object_id, row, row["action"], all_netobj2pro, all_subject2pro, all_file2pro)
        add_node_properties(nodes, object_id, properties1)

        edge = (actor_id, object_id)
        edges.append(edge)
        relations[edge] = action

        ## 构建图
        # 点不重复添加
        actor_idx = get_or_add_node(G, actor_id, ObjectType[row['actor_type']].value, properties)
        object_idx = get_or_add_node(G, object_id, ObjectType[row['object']].value, properties)
        # 边也不重复添加
        add_edge_if_new(G, actor_idx, object_idx, action)

    features, edge_index, index_map, relations_index = [], [[], []], {}, {}
    for node_id, props in nodes.items():
        features.append(props)
        index_map[node_id] = len(features) - 1

    update_edge_index(edges, edge_index, index_map, relations, relations_index)

    return features, edge_index, list(index_map.keys()), relations_index, G


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

def add_attributes(d, p):
    f = open(p)
    # data = [json.loads(x) for x in f if "EVENT" in x]
    # for test
    data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x and i < 300]
    info = []
    for x in data:
        try:
            action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
        except:
            action = ''
        try:
            actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            actor = ''
        try:
            obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject'][
                'com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            obj = ''
        try:
            timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
        except:
            timestamp = ''
        try:
            cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['cmdLine']
        except:
            cmd = ''
        try:
            path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
        except:
            path = ''
        try:
            path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
        except:
            path2 = ''
        try:
            obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2'][
                'com.bbn.tc.schema.avro.cdm18.UUID']
            info.append({'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp, 'exec': cmd,
                         'path': path2})
        except:
            pass

        info.append(
            {'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path})

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()


def collect_nodes_from_log(paths):
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                # --- NetFlowObject ---
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line
                        )[0]
                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]
                        nodeproperty = f"{srcaddr},{srcport},{dstaddr},{dstport}"
                        netobj2pro[nodeid] = nodeproperty
                    except:
                        pass

                # --- Subject ---
                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line:
                    try:
                        res = re.findall(
                            'Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"',
                            line
                        )[0]
                        nodeid = res[0]
                        cmdLine = res[2]
                        tgid = res[4]
                        try:
                            path_str = re.findall('"path":"(.*?)"', line)[0]
                            path = path_str
                        except:
                            path = "null"
                        nodeProperty = f"{cmdLine},{tgid},{path}"
                        subject2pro[nodeid] = nodeProperty
                    except:
                        pass

                # --- FileObject ---
                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line:
                    try:
                        res = re.findall(
                            'FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"',
                            line
                        )[0]
                        nodeid = res[0]
                        filepath = res[2]
                        nodeproperty = filepath
                        file2pro[nodeid] = nodeproperty
                    except:
                        pass

    return netobj2pro, subject2pro, file2pro


def collect_edges_from_log(d, paths):
    info = []
    for p in paths:
        with open(p) as f:
            # TODO
            # for test: 只取每个文件前300条包含"EVENT"的
            data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x and i < 1000]
            # data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x ]
        for x in data:
            try:
                action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
            except:
                action = ''
            try:
                actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
            except:
                actor = ''
            try:
                obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject'][
                    'com.bbn.tc.schema.avro.cdm18.UUID']
            except:
                obj = ''
            try:
                timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
            except:
                timestamp = ''
            try:
                cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['cmdLine']
            except:
                cmd = ''
            try:
                path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
            except:
                path = ''
            try:
                path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
            except:
                path2 = ''
            try:
                obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2'][
                    'com.bbn.tc.schema.avro.cdm18.UUID']
                info.append({
                    'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp,
                    'exec': cmd, 'path': path2
                })
            except:
                pass

            info.append({
                'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp,
                'exec': cmd, 'path': path
            })

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()


