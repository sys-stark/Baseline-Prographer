# =================训练=========================
import sys
import os
# 这个代码块修复了导入路径问题
# 它将父目录（也就是项目根目录）添加到了 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.match.test_model import test_model
from process.partition import detect_communities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取测试数据集
# data_handler = get_handler("atlas", False)
data_handler,_ = ("atlas", False)
data_handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G_snapshots = data_handler.build_graph()
print(f"总共生成了 {len(G_snapshots)} 个快照。")
#嵌入构造特征向量
embedder_class = get_embedder_by_name("prographer")
embedder = embedder_class(G_snapshots)
embedder.train()
snapshot_embeddings = embedder.get_snapshot_embeddings()
rsg_embeddings, rsg_vocab = embedder.get_rsg_embeddings()
test_model(G_snapshots,snapshot_embeddings,rsg_embeddings,rsg_vocab)
