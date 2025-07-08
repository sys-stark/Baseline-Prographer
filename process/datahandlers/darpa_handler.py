import os

import igraph as ig
import re
import pandas as pd
import json
from .common import collect_json_paths, collect_label_paths
from .base import BaseProcessor
from .type_enum import ObjectType
from .common import merge_properties, collect_dot_paths,extract_properties,add_node_properties,get_or_add_node,add_edge_if_new,update_edge_index


class DARPAHandler(BaseProcessor):
    def load(self):
        json_map = collect_json_paths(self.base_path)
        label_map = collect_label_paths(self.base_path)
        for scene, category_data in json_map.items():
            # TODO: for test
            if scene != "theia33":
                continue
            if self.train == False:
                label_file = open(label_map[scene])
                print(f"正在处理: 场景={scene}, label={label_map[scene]}")
                self.all_labels.extend([
                    line.strip() for line in label_file.read().splitlines() if line.strip()
                ])
            for category, json_files in category_data.items():
                #  训练只处理良性类别
                if self.train and category != "benign":
                    continue

                print(f"正在处理: 场景={scene}, 类别={category}, 文件={json_files}")
                scene_category = f"/{scene}_{category}.txt"
                f = open(self.base_path + scene_category)
                self.total_loaded_bytes += os.path.getsize(self.base_path + scene_category)
                # 训练分隔
                data = f.read().split('\n')
                # TODO:
                # data = [line.split('\t') for line in data]
                # for test
                data = [line.split('\t') for line in data[:10000]]
                df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
                df = df.dropna()
                df.sort_values(by='timestamp', ascending=True, inplace=True)

                # 形成一个更完整的视图
                netobj2pro, subject2pro, file2pro = collect_nodes_from_log(json_files)
                df = collect_edges_from_log(df, json_files)

                if self.train:
                    # 只取良性前80%训练
                    num_rows = int(len(df) * 0.9)
                    df = df.iloc[:num_rows]
                    self.all_dfs.append(df)
                else:
                    # 数据选择逻辑
                    if category == "benign":
                        # 取后10%
                        num_rows = int(len(df) * 0.9)
                        df = df.iloc[num_rows:]
                        self.all_dfs.append(df)
                    elif category == "malicious":
                        # 使用全部
                        self.all_dfs.append(df)
                        pass
                    else:
                        continue
                merge_properties(netobj2pro, self.all_netobj2pro)
                merge_properties(subject2pro, self.all_subject2pro)
                merge_properties(file2pro, self.all_file2pro)
        # 训练用的数据集
        use_df = pd.concat(self.all_dfs, ignore_index=True)
        self.use_df = use_df.drop_duplicates()



    def build_graph(self):
        """成图+捕捉特征语料+简化策略这里添加"""
        G = ig.Graph(directed=True)
        nodes, edges, relations = {}, [], {}

        for _, row in self.use_df.iterrows():
            action = row["action"]

            actor_id = row["actorID"]
            properties = extract_properties(actor_id, row, row["action"], self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes, actor_id, properties)

            object_id = row["objectID"]
            properties1 = extract_properties(object_id, row, row["action"], self.all_netobj2pro, self.all_subject2pro,
                                             self.all_file2pro)
            add_node_properties(nodes, object_id, properties1)

            edge = (actor_id, object_id)
            edges.append(edge)
            relations[edge] = action

            ## 构建图
            # 点不重复添加
            actor_idx = get_or_add_node(G, actor_id, ObjectType[row['actor_type']].value, properties)
            object_idx = get_or_add_node(G, object_id, ObjectType[row['object']].value, properties)
            # 标注label
            print(f"actor_id{actor_id}")
            print(f"object_id{object_id} value{int(object_id in self.all_labels)}")
            G.vs[actor_idx]["label"] = int(actor_id in self.all_labels)
            G.vs[object_idx]["label"] = int(object_id in self.all_labels)
            # 边也不重复添加
            add_edge_if_new(G, actor_idx, object_idx, action)

        features, edge_index, index_map, relations_index = [], [[], []], {}, {}
        for node_id, props in nodes.items():
            features.append(props)
            index_map[node_id] = len(features) - 1

        update_edge_index(edges, edge_index, index_map, relations, relations_index)

        return features, edge_index, list(index_map.keys()), relations_index, G


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
            data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x and i < 10000]
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

