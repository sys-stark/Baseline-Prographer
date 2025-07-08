import igraph as ig
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GNNExplainer
from torch_geometric.datasets import Planetoid


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def convert_igraph_to_pyg(igraph_graph):
    """将 igraph 图转换为 PyTorch Geometric 格式"""
    edge_list = torch.tensor(igraph_graph.get_edgelist(), dtype=torch.long).t().contiguous()
    x = torch.eye(igraph_graph.vcount(), dtype=torch.float)  # 简单的 one-hot 编码特征
    return Data(x=x, edge_index=edge_list)

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

explainer = GNNExplainer(model, epochs=200)


#
# node_idx = 0  # 要解释的节点索引
# node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
# print("node_feature_mask", node_feat_mask)
# print("edge_mask", edge_mask)
#
# plt.figure()
# ax, G = explainer.visualize_subgraph(node_idx, data.edge_index, edge_mask, y=data.y)
# output_path = "gnn_explainer_output.png"
# plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存高分辨率图像




graph_feat_mask, graph_edge_mask = explainer.explain_graph(data.x, data.edge_index)
# **打印解释结果**
print(" 整个图的特征重要性:", graph_feat_mask)
print(" 整个图的边重要性:", graph_edge_mask)
# **可视化整个图的解释**
plt.figure()
ax, G = explainer.visualize_subgraph(-1, data.edge_index.cpu(), graph_edge_mask.cpu())
graph_output_path = "gnn_explainer_graph_output.png"
plt.savefig(graph_output_path, dpi=300, bbox_inches='tight')
print(f" 图的解释图已保存到 {graph_output_path}")

# *** 最重要的点和边
node_importance = torch.zeros(data.x.shape[0])
for idx, importance in enumerate(graph_edge_mask.cpu().detach().numpy()):
    src, dst = data.edge_index[:, idx]  # 获取边的两个端点
    node_importance[src] += importance  # 源节点
    node_importance[dst] += importance  # 目标节点
sorted_nodes = sorted(enumerate(node_importance.numpy()), key=lambda x: x[1], reverse=True)
# **打印最重要的前 5 个节点**
print("最重要的节点（前 5）：")
for idx, importance in sorted_nodes[:5]:
    print(f"节点 {idx} → 重要性: {importance:.4f}")
edge_importance = graph_edge_mask.cpu().detach().numpy()
sorted_edges = sorted(enumerate(edge_importance), key=lambda x: x[1], reverse=True)
print("最重要的边（前 5）：")
for idx, importance in sorted_edges[:5]:  # 取前 5 条
    print(f"边 {data.edge_index[:, idx].tolist()} → 重要性: {importance:.4f}")