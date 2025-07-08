# =================训练=========================
import os
import pandas as pd
import torch
from process.match.deephunter.make_graph import prepare_graph_new, collect_edges_from_log, \
    collect_nodes_from_log
from process.match.deephunter.test_model import test_model
from process.model import EpochLogger, EpochSaver, infer
from process.partition import detect_communities
from gensim.models import Word2Vec


def merge_properties(src_dict, target_dict):
    for k, v in src_dict.items():
        if k not in target_dict:
            target_dict[k] = v

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = EpochLogger()
saver = EpochSaver()

# 加载一个数据集
base_path = "../../../data_files/theia"
json_map = collect_json_paths(base_path)

all_dfs = []
all_netobj2pro = {}  # 网络对象 UUID → 属性字符串
all_subject2pro = {}  # 进程 UUID → 属性字符串
all_file2pro = {}  # 文件 UUID → 属性字符串

for scene, category_data in json_map.items():
    for category, json_files in category_data.items():
        # TODO: for test
        if scene != "theia33":
            continue

        print(f"正在处理: 场景={scene}, 类别={category}, 文件={json_files}")
        scene_category = f"/{scene}_{category}.txt"
        f = open(base_path + scene_category)

        # 训练分隔
        data = f.read().split('\n')
        # TODO:
        # data = [line.split('\t') for line in data]
        # for test
        data = [line.split('\t') for line in data[:1000]]
        df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
        df = df.dropna()
        df.sort_values(by='timestamp', ascending=True, inplace=True)

        # 形成一个更完整的视图
        netobj2pro, subject2pro, file2pro = collect_nodes_from_log(json_files)
        df = collect_edges_from_log(df, json_files)

        # 数据选择逻辑
        if category == "benign":
            # 取后10%
            num_rows = int(len(df) * 0.9)
            df = df.iloc[num_rows:]
            all_dfs.append(df)
        elif category == "malicious":
            # 使用全部
            all_dfs.append(df)
            pass
        else:
            continue

        merge_properties(netobj2pro, all_netobj2pro)
        merge_properties(subject2pro, all_subject2pro)
        merge_properties(file2pro, all_file2pro)

# 测试用的数据集
test_all_df = pd.concat(all_dfs, ignore_index=True)
test_all_df = test_all_df.drop_duplicates()
test_all_df.to_csv("test_all_df.txt", sep='\t', index=False)

# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G = prepare_graph_new(test_all_df, all_netobj2pro, all_subject2pro, all_file2pro)

# 大图分割
communities = detect_communities(G)

sentences = []
for (src, dst), rel in relations.items():
    if src < len(features) and dst < len(features):
        src_attrs = features[src]
        dst_attrs = features[dst]
        sentence = src_attrs + [rel] + dst_attrs
        sentences.append(sentence)
# 用于编码节点属性的属性嵌入网络word2Vec
word2vec = Word2Vec(sentences=sentences, vector_size=30, window=5, min_count=1, workers=8, epochs=300,
                    callbacks=[saver, logger])

# 点
node_name_list = [v["name"] for v in G.vs]
node_feature_list = [str(features[mapp.index(name)]) for name in node_name_list]
node_embeddings = {}
for node_name, node_feature in zip(node_name_list, node_feature_list):
    embedding = infer(node_feature, "word2vec_theia_E3.model")
    node_embeddings[node_name] = embedding  # 使用node_name作为键存储embedding
    print(f"Node '{node_name}' embedding: {embedding[:5]}")  # 只显示前5维的embedding
# 边
edge_list = [edge['actions'] if 'actions' in edge.attributes() else "undefined_relation" for edge in G.es]
edge_embeddings = {}
for relation in edge_list:
    embedding =infer([relation], "word2vec_theia_E3.model")
    edge_embeddings[relation] = embedding
    print(f"Relation '{relation}' embedding: {embedding[:5]}")  # 打印前5维

# 模型测试
test_model(G, communities, node_embeddings, edge_embeddings)

