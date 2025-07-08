from process.match.deephunter.graphembnet import GraphEmbeddingNet
from process.match.deephunter.graphembnet import GraphPropLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """Compute the dot product similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise dot product similarity.
    """
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), 1e-12)))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}



class GraphMatchingNet(GraphEmbeddingNet):
    """Graph matching net.

    This class uses graph matching layers instead of the simple graph prop layers.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 similarity='dotproduct',
                 prop_type='embedding'):
        super(GraphMatchingNet, self).__init__(
            encoder,
            aggregator,
            node_state_dim,
            edge_state_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            n_prop_layers,
            share_prop_params=share_prop_params,
            edge_net_init_scale=edge_net_init_scale,
            node_update_type=node_update_type,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm,
            # TODO
            layer_class=GraphPropLayer,
            prop_type=prop_type,
        )
        # TODO
        self._similarity = similarity


class NTN(nn.Module):
    def __init__(self, input_dim, slice_num):
        super(NTN, self).__init__()
        self.slice_num = slice_num
        self.input_dim = input_dim

        # 张量交互项参数：K 个 D×D 的关系矩阵
        self.W = nn.Parameter(torch.Tensor(slice_num, input_dim, input_dim))
        # 线性项参数：K 个 [2D → 1] 的向量
        self.V = nn.Parameter(torch.Tensor(slice_num, 2 * input_dim))
        # 每个 slice 的偏置
        self.b = nn.Parameter(torch.Tensor(slice_num))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.b)

    def forward(self, h1, h2):
        """
        输入：
            h1: [B, D]  第一个图的表示
            h2: [B, D]  第二个图的表示
        输出：
            [B, K]，表示每对图在 K 个关系通道下的交互得分
        """
        # 张量双线性交互项：每个 slice k 执行 h1^T W_k h2
        bilinear_term = torch.einsum('bi,kij,bj->bk', h1, self.W, h2)  # [B, K]

        # 线性项：V_k × [h1; h2] + b_k
        linear_input = torch.cat([h1, h2], dim=1)  # [B, 2D]
        linear_term = F.linear(linear_input, self.V, self.b)  # [B, K]

        return torch.tanh(bilinear_term + linear_term)

class GraphMatchingScorer(nn.Module):
    def __init__(self, embed_dim, ntn_slices=16, dnn_hidden_dim=64):
        super(GraphMatchingScorer, self).__init__()
        self.ntn = NTN(embed_dim, slice_num=ntn_slices)
        self.dnn = nn.Sequential(
            nn.Linear(ntn_slices, dnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dnn_hidden_dim, 1)  # 输出匹配分数 s
        )

    def forward(self, h1, h2):
        ntn_out = self.ntn(h1, h2)
        s = self.dnn(ntn_out).squeeze(-1)  # 输出 [batch]
        return s