import igraph as ig
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GNNExplainer


### 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None, edge_attr=None, dropout_rate=0.0):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


### ** 将 `igraph` 图转换为 `PyG` 格式**
def convert_igraph_to_pyg(igraph_graph):
    """将 igraph 图转换为 PyTorch Geometric 格式"""
    edge_list = torch.tensor(igraph_graph.get_edgelist(), dtype=torch.long).t().contiguous()
    x = torch.eye(igraph_graph.vcount(), dtype=torch.float)  # one-hot 编码特征
    return Data(x=x, edge_index=edge_list)


### ** 创建 `igraph` 图**
G = ig.Graph(directed=True)
G.add_vertices(["A", "B", "C", "D", "E", "F", "G", "H"])
G.add_edges([
    ("A", "B"), ("B", "C"), ("C", "D"),
    ("E", "F"), ("F", "G"), ("G", "H"),
    ("D", "E")  # 连接两个社区的桥接边
])
data = convert_igraph_to_pyg(G)

### ** 处理 `train_mask` 和 `data.y`
num_nodes = data.x.shape[0]
num_classes = 2  # 假设是二分类任务

# 生成随机标签（如果 data.y 为空）
if not hasattr(data, "y") or data.y is None:
    data.y = torch.randint(0, num_classes, (num_nodes,))

# 生成 `train_mask`（80% 训练，20% 其他）
train_ratio = 0.8
num_train = int(train_ratio * num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[:num_train] = True  # 80% 用于训练
train_mask = train_mask[torch.randperm(num_nodes)]  # 随机打乱
data.train_mask = train_mask  # 添加到 `data`

### **训练 GCN**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
model = GCN(data.x.shape[1], 16, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("训练 GCN ...")
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])  # 修正损失计算
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

print("GCN 训练完成！")

### ** 解释整个图**
explainer = GNNExplainer(model, epochs=200)
graph_feat_mask, graph_edge_mask = explainer.explain_graph(data.x, data.edge_index)

# **打印解释结果**
print(" 整个图的特征重要性:", graph_feat_mask)
print(" 整个图的边重要性:", graph_edge_mask)

# **可视化整个图**
plt.figure()
ax, G = explainer.visualize_subgraph(-1, data.edge_index.cpu(), graph_edge_mask.cpu(), y=data.y)
plt.title("Graph Importance (GNNExplainer)")
plt.savefig("gnn_explainer_graph_output.png", dpi=300, bbox_inches="tight")
print("图的解释图已保存到 gnn_explainer_graph_output.png")


### ** 计算最重要的节点和边**
def find_important_nodes_and_edges(graph_edge_mask, edge_index):
    """计算最重要的节点和边"""

    # **计算节点的重要性（累加与该节点相连的边的分数）**
    node_importance = torch.zeros(data.x.shape[0])
    for idx, importance in enumerate(graph_edge_mask.cpu().detach().numpy()):
        src, dst = edge_index[:, idx]  # 获取边的两个端点
        node_importance[src] += importance  # 源节点
        node_importance[dst] += importance  # 目标节点

    # **按重要性排序**
    sorted_nodes = sorted(enumerate(node_importance.numpy()), key=lambda x: x[1], reverse=True)
    sorted_edges = sorted(enumerate(graph_edge_mask.cpu().detach().numpy()), key=lambda x: x[1], reverse=True)

    # **打印最重要的前 5 个节点**
    print(" 最重要的节点（前 5）：")
    for idx, importance in sorted_nodes[:5]:
        print(f"节点 {idx} → 重要性: {importance:.4f}")

    # **打印最重要的前 5 条边**
    print(" 最重要的边（前 5）：")
    for idx, importance in sorted_edges[:5]:
        print(f"边 {edge_index[:, idx].tolist()} → 重要性: {importance:.4f}")


find_important_nodes_and_edges(graph_edge_mask, data.edge_index)