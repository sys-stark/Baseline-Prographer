# =================训练=========================
import sys
import os
# 这个代码块修复了导入路径问题
# 它将父目录（也就是项目根目录）添加到了 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.match.match import train_model
from process.partition import detect_communities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据集
# data_handler = get_handler("atlas")
data_handler = get_handler("atlas", True)

# 加载数据
data_handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G_snapshots,_,_ = data_handler.build_graph()
print(f"总共生成了 {len(G_snapshots)} 个快照。")
#print("features:", features)
#print("edges:", edges)
#print("mapp:", mapp)
#print("relations:", relations)
#for i, snapshot in enumerate(G_snapshots):
# print(f"\n--- 快照 {i+1} ---")
# print(snapshot) # 单独打印每一个 snapshot 对象
#嵌入构造特征向量
embedder_class = get_embedder_by_name("prographer")
embedder = embedder_class(G_snapshots)
embedder.train()
snapshot_embeddings = embedder.get_snapshot_embeddings()
rsg_embeddings, rsg_vocab = embedder.get_rsg_embeddings()
print("\n--- Encoder process finished ---")
print(f"已生成快照嵌入序列，形状为: {snapshot_embeddings.shape}")
print(f"已生成RSG嵌入矩阵，形状为: {rsg_embeddings.shape}")
print(f"RSG词汇表大小: {len(rsg_vocab)}")
# 模型训练
# 匹配
train_model(G_snapshots,snapshot_embeddings,rsg_embeddings,rsg_vocab)