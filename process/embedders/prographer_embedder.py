# The updating process follows the skipgram model [39] used by doc2vec....For training efficiency, we apply negative sampling [40] like prior works

import torch
import torch.nn as nn
import torch.optim as optim
import igraph as ig
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from .base import GraphEmbedderBase

class ProGrapherEmbedder(GraphEmbedderBase):
    def __init__(self, snapshot_sequence,
                 # --- Encoder (Graph2Vec) Parameters from Paper ---
                 embedding_dim=256,
                 wl_depth=4,
                 neg_samples=15,
                 # --- Training Hyperparameters ---
                 learning_rate=1e-3,
                 epochs=1,
                 weight_decay=1e-5
                 ):

        super().__init__(snapshot_sequence,features=None,mapp=None)
        self.snapshot_sequence = self.G

        # --- Assign all encoder-specific parameters ---
        self.embedding_dim = embedding_dim
        self.wl_depth = wl_depth
        self.neg_samples = neg_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay

        # --- Internal state variables for the encoder ---
        self.rsg_vocab = {} # 从 RSG 字符串到索引的映射
        self.snapshot_embeddings_layer = None
        self.rsg_embeddings_layer = None

        ### GPU MODIFICATION 1: 定义设备 ###
        # 在类的初始化方法中定义device，这样类的所有方法都可以访问它。
        # 这使得代码在有GPU的环境中自动使用GPU，否则回退到CPU。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- ProGrapherEmbedder will use device: {self.device} ---")


    # --- 以下的静态方法 _get_neighbor_info 和 generate_rsg 不需要修改 ---
    # --- 因为它们处理的是igraph对象和字符串，这些操作在CPU上进行效率更高 ---
    @staticmethod
    def _get_neighbor_info(graph, edge, node_idx):
        if edge.source == node_idx:
            return edge.target
        return edge.source

    @staticmethod
    def generate_rsg(graph, node_idx, depth):
        if depth == 0:
            return str(graph.vs[node_idx]['type'])

        prev_rsg = ProGrapherEmbedder.generate_rsg(graph, node_idx, depth - 1)
        incident_edges = graph.es[graph.incident(node_idx, mode="all")]
        if not incident_edges:
            return prev_rsg

        neighbor_info_parts = []
        for edge in incident_edges:
            try:
                edge_type = str(edge['actions'])
            except (KeyError, TypeError):
                edge_type = "UNKNOWN"
            neighbor_idx = ProGrapherEmbedder._get_neighbor_info(graph, edge, node_idx)
            neighbor_rsg = ProGrapherEmbedder.generate_rsg(graph, neighbor_idx, depth - 1)
            neighbor_info_parts.append(f"{edge_type}:{neighbor_rsg}")

        sorted_neighbor_info = sorted(neighbor_info_parts)
        return f"{prev_rsg}-({'_'.join(sorted_neighbor_info)})"

    # --- _build_vocabulary 也不需要修改，它在CPU上处理数据构建词汇表 ---
    def _build_vocabulary(self):
        print("Building RSG vocabulary from all snapshots...")
        all_rsgs = set()
        for snapshot in tqdm(self.snapshot_sequence, desc="Processing Snapshots for Vocab"):
            for v_idx in range(len(snapshot.vs)):
                for d in range(self.wl_depth + 1):
                    all_rsgs.add(ProGrapherEmbedder.generate_rsg(snapshot, v_idx, d))
        self.rsg_vocab = {rsg: i for i, rsg in enumerate(sorted(list(all_rsgs)))}
        print(f"Vocabulary built with {len(self.rsg_vocab)} unique RSGs.")

    def train(self):
        """
        训练 Graph2Vec 模型，学习快照和RSG的嵌入。
        (已修改为在GPU上训练)
        """
        print("--- Training ProGrapher Encoder (Graph2Vec) ---")
        if not self.snapshot_sequence:
            print("Warning: No snapshots to train on.")
            return

        # CPU密集型任务，保留在CPU上
        self._build_vocabulary()
        num_snapshots = len(self.snapshot_sequence)
        num_rsgs = len(self.rsg_vocab)

        if num_rsgs == 0:
            print("Warning: RSG vocabulary is empty. Cannot train.")
            return

        self.snapshot_embeddings_layer = nn.Embedding(num_snapshots, self.embedding_dim)
        self.rsg_embeddings_layer = nn.Embedding(num_rsgs, self.embedding_dim)
        nn.init.xavier_uniform_(self.snapshot_embeddings_layer.weight)
        nn.init.xavier_uniform_(self.rsg_embeddings_layer.weight)

        ### GPU MODIFICATION 2: 将模型和损失函数移动到设备 ###
        # 将两个嵌入层（即我们的模型）的参数和缓冲区移动到指定的device（GPU或CPU）。
        self.snapshot_embeddings_layer.to(self.device)
        self.rsg_embeddings_layer.to(self.device)
        # 损失函数通常是无状态的，但将其移动到设备上是一个好习惯，以防它内部有可学习的参数。
        criterion = nn.BCEWithLogitsLoss().to(self.device)

        optimizer = optim.Adam(
            list(self.snapshot_embeddings_layer.parameters()) + list(self.rsg_embeddings_layer.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            total_loss = 0
            num_updates = 0
            shuffled_indices = np.random.permutation(num_snapshots)

            for snapshot_idx in tqdm(shuffled_indices, desc=f"Encoder Epoch {epoch+1}/{self.epochs}", leave=False):
                snapshot = self.snapshot_sequence[snapshot_idx]

                # 这部分逻辑在CPU上运行更快，因为它涉及大量Python循环和字典查找
                positive_rsg_ids = {
                    self.rsg_vocab[ProGrapherEmbedder.generate_rsg(snapshot, v_idx, d)]
                    for v_idx in range(len(snapshot.vs))
                    for d in range(self.wl_depth + 1)
                    if ProGrapherEmbedder.generate_rsg(snapshot, v_idx, d) in self.rsg_vocab
                }
                if not positive_rsg_ids: continue

                for rsg_id in positive_rsg_ids:
                    # 负采样仍然在CPU上用numpy完成，速度很快
                    neg_sample_ids = []
                    while len(neg_sample_ids) < self.neg_samples:
                        sample = np.random.randint(0, num_rsgs)
                        if sample != rsg_id and sample not in positive_rsg_ids:
                            neg_sample_ids.append(sample)

                    ### GPU MODIFICATION 3: 将数据移动到设备 ###
                    # 这是最关键的一步。在将数据送入模型之前，必须将其从CPU内存转移到GPU显存。
                    # 我们使用 .to(self.device) 来完成这个操作。

                    # 1. 目标ID（1个正样本 + N个负样本）
                    target_ids = torch.LongTensor([rsg_id] + neg_sample_ids).to(self.device)
                    # 2. 对应的标签
                    labels = torch.FloatTensor([1.0] + [0.0] * self.neg_samples).to(self.device)
                    # 3. 当前快照的ID
                    snapshot_id_tensor = torch.LongTensor([snapshot_idx]).to(self.device)

                    # --- 以下是PyTorch的计算图，现在将在GPU上执行 ---
                    snapshot_vec = self.snapshot_embeddings_layer(snapshot_id_tensor)
                    rsg_vecs = self.rsg_embeddings_layer(target_ids)

                    logits = torch.sum(snapshot_vec * rsg_vecs, dim=1)

                    loss = criterion(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # loss.item() 会自动将GPU上的单个标量值拷贝回CPU，以便累加
                    total_loss += loss.item()
                    num_updates += 1

            avg_loss = total_loss / num_updates if num_updates > 0 else 0
            print(f"Encoder Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.6f}")

        print("\nEncoder training complete.")

    # --- 以下的 `get_...` 和 `embed_...` 方法需要确保从GPU取回数据 ---
    # --- 你的原始代码已经正确使用了 .detach().cpu().numpy()，所以无需修改 ---
    def get_snapshot_embeddings(self):
        print("Retrieving all snapshot embeddings...")
        if self.snapshot_embeddings_layer is None:
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        # .weight 在GPU上，所以需要用 .detach().cpu().numpy() 将其取回为numpy数组
        return self.snapshot_embeddings_layer.weight.detach().cpu().numpy()

    def get_rsg_embeddings(self):
        print("Retrieving all RSG embeddings and vocabulary...")
        if self.rsg_embeddings_layer is None:
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        # 同上，从GPU取回数据
        rsg_embeddings = self.rsg_embeddings_layer.weight.detach().cpu().numpy()
        return rsg_embeddings, self.rsg_vocab

    # --- embed_nodes 和 embed_edges 方法使用numpy在CPU上操作，无需修改 ---
    # --- 它们依赖于上面两个已经将数据转为numpy数组的方法，所以是安全的。 ---
    def embed_nodes(self):
        """
        为 'train.py' 提供节点嵌入。
        此方法现在依赖于 get_rsg_embeddings()，后者会处理GPU到CPU的数据转换。
        """
        print("Generating node embeddings using the final snapshot's structure and globally trained RSG embeddings...")

        # 为了使用嵌入，我们首先需要从GPU获取它们
        # 注意：这里我们假设模型已经训练完毕
        # 在原代码中，这里引用了 self.rsg_embeddings，我们改为直接调用获取方法
        try:
            rsg_embeddings_np, rsg_vocab_map = self.get_rsg_embeddings()
        except RuntimeError as e:
            print(e)
            return {}

        # 创建一个从 rsg_str 到 embedding 的映射，方便查找
        rsg_str_to_emb = {rsg: rsg_embeddings_np[idx] for rsg, idx in rsg_vocab_map.items()}

        node_embeddings = {}

        if not self.snapshot_sequence:
            print("Warning: No snapshots available to generate node embeddings.")
            return {}

        final_snapshot = self.snapshot_sequence[-1]

        for v in final_snapshot.vs:
            node_name = v['name']
            rsg_str = ProGrapherEmbedder.generate_rsg(final_snapshot, v.index, self.wl_depth)

            # 使用我们创建的查找表获取嵌入
            embedding = rsg_str_to_emb.get(rsg_str, np.zeros(self.embedding_dim))
            node_embeddings[node_name] = embedding

        for node_name in self.mapp.keys():
            if node_name not in node_embeddings:
                node_embeddings[node_name] = np.zeros(self.embedding_dim)

        print(f"Generated {len(node_embeddings)} node embeddings.")
        return node_embeddings

    def embed_edges(self):
        """
        为 'train.py' 提供边嵌入。
        此方法依赖于 embed_nodes()，它在CPU上运行，所以这里也无需修改。
        """
        print("Generating edge embeddings based on endpoint node embeddings...")

        node_embeddings = self.embed_nodes()
        if not node_embeddings:
            print("Warning: Node embeddings are empty, cannot generate edge embeddings.")
            return {}

        edge_embeddings = {}
        # 假设 self.global_edges 已经填充
        if not hasattr(self, 'global_edges') or not self.global_edges:
            # 在Base类中构建全局边
            self._build_global_edges()

        for src_name, tgt_name, action in self.global_edges:
            source_emb = node_embeddings.get(src_name, np.zeros(self.embedding_dim))
            target_emb = node_embeddings.get(tgt_name, np.zeros(self.embedding_dim))
            edge_emb = (source_emb + target_emb) / 2.0
            edge_embeddings[action] = edge_emb

        print(f"Generated embeddings for {len(edge_embeddings)} unique edge types.")
        return edge_embeddings