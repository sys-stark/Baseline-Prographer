import io
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
# --- 1. 设置和导入 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process.datahandlers import get_handler
from process.embedders import get_embedder_by_name, ProGrapherEmbedder  # 直接导入ProGrapherEmbedder类

# --- 2. 模型定义 ---
class AnomalyDetector(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, kernel_sizes, num_filters):
        super(AnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim * 2,
                out_channels=num_filters,
                kernel_size=k
            ) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embedding_dim)

    def forward(self, sequence_embeddings):
        lstm_out, _ = self.lstm(sequence_embeddings)
        conv_in = lstm_out.permute(0, 2, 1)

        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(conv_in))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)

        concatenated = torch.cat(pooled_outputs, dim=1)
        dropped_out = self.dropout(concatenated)
        predicted_embedding = self.fc(dropped_out)
        return predicted_embedding

# --- 3. 超参数设置 ---
SEQUENCE_LENGTH_L = 12
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
NUM_LAYERS = 5
DROPOUT_RATE = 0.2
KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 100
DETECTION_THRESHOLD = 0.01
TOP_K_INDICATORS = 5  # 要报告的顶级异常RSG数量
WL_DEPTH = 4  # Weisfeiler-Lehman子树深度
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- 4.1 预测函数（返回预测标签、嵌入差异和位置）---
def predict_anomalous_snapshots(snapshot_embeddings, model_path):
    """加载模型并预测异常快照"""
    detector_model = AnomalyDetector(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
        kernel_sizes=KERNEL_SIZES,
        num_filters=NUM_FILTERS
    ).to(device)

    detector_model.load_state_dict(torch.load(model_path, map_location=device))
    detector_model.eval()

    tensor = torch.tensor(snapshot_embeddings, dtype=torch.float32)
    snapshot_pred_labels = np.zeros(len(tensor), dtype=int)
    # 存储每个位置的差异向量（异常检测的重要信息）
    diff_vectors = {}

    with torch.no_grad():
        for i in tqdm(range(len(tensor) - SEQUENCE_LENGTH_L),
                      desc="检测快照序列",
                      leave=True,
                      unit="snapshot"):
            sequence = tensor[i:i+SEQUENCE_LENGTH_L].unsqueeze(0).to(device)
            target = tensor[i+SEQUENCE_LENGTH_L].unsqueeze(0).to(device)
            prediction = detector_model(sequence).squeeze(0)

            # 计算误差和差异向量
            error = torch.nn.functional.mse_loss(prediction, target).item()
            diff_vector = (prediction - target).cpu().numpy()

            # 如果误差大于阈值，标记为异常
            if error > DETECTION_THRESHOLD:
                snapshot_pred_labels[i+SEQUENCE_LENGTH_L] = 1
                # 存储差异向量和位置
                diff_vectors[i+SEQUENCE_LENGTH_L] = {
                    "position": i+SEQUENCE_LENGTH_L,
                    "error": error,
                    "diff_vector": diff_vector,
                    "real_embedding": target.cpu().numpy(),    # 修复：移除 .squeeze(0)
                    "pred_embedding": prediction.cpu().numpy() # 修复：移除 .squeeze(0)
                }

    return snapshot_pred_labels, diff_vectors

# --- 4.2 获取真实标签函数 ---
def get_true_snapshot_labels(snapshots):
    """获取每个快照的真实标签（是否包含恶意节点）"""
    true_labels = []
    for snapshot in snapshots:
        malicious_nodes = any(v['label'] == 1 for v in snapshot.vs)
        true_labels.append(1 if malicious_nodes else 0)
    return np.array(true_labels)

# --- 5.1 关键指标生成器 ---
def generate_key_indicators(all_snapshots, diff_vectors, rsg_embeddings, rsg_vocab):
    """生成关键异常指标（可疑RSG排名）"""
    print("\n" + "="*50)
    print(" 关键指标生成器 - 异常RSG排名")
    print("="*50)

    if not diff_vectors:
        print("未检测到任何异常快照，跳过指标生成")
        return

    total_anomalies = len(diff_vectors)
    progress = tqdm(diff_vectors.items(), total=total_anomalies, desc="分析异常快照")

    for idx, anomaly_info in progress:
        snapshot = all_snapshots[idx]
        diff_vector = anomaly_info["diff_vector"]

        # 确保 diff_vector 是 1D 向量
        if diff_vector.ndim > 1:
            diff_vector = diff_vector.squeeze()

        rsg_scores = defaultdict(float)
        rsg_count = 0

        for v_idx in range(len(snapshot.vs)):
            for d in range(WL_DEPTH + 1):
                rsg_str = ProGrapherEmbedder.generate_rsg(snapshot, v_idx, d)

                if rsg_str in rsg_vocab:
                    rsg_id = rsg_vocab[rsg_str]
                    rsg_vec = rsg_embeddings[rsg_id]

                    # 确保 rsg_vec 是 1D 向量
                    if rsg_vec.ndim > 1:
                        rsg_vec = rsg_vec.squeeze()

                    # 使用 vdot 计算点积（返回标量）
                    score = np.abs(np.vdot(diff_vector, rsg_vec))

                    rsg_scores[rsg_str] = max(rsg_scores[rsg_str], score)
                    rsg_count += 1

        if not rsg_scores:
            progress.set_postfix(snapshot=f"{idx}", info=f"未找到RSG")
            continue

        sorted_rsgs = sorted(rsg_scores.items(), key=lambda x: x[1], reverse=True)

        # 现在 sorted_rsgs[0][1] 应该是标量值
        progress.set_postfix(snapshot=f"{idx}", score=f"{sorted_rsgs[0][1]:.4f}")

        print(f"\n异常快照 {idx} (检测差异: {anomaly_info['error']:.6f})")
        print(f"  - 总RSG数量: {rsg_count}")
        print(f"  - Top-{TOP_K_INDICATORS} 可疑 RSG:")

        for i, (rsg, score) in enumerate(sorted_rsgs[:TOP_K_INDICATORS]):
            print(f"    {i+1}. {rsg} (可疑度: {score:.6f})")

    print("="*50 + "\n")

# --- 5.2 核心评估逻辑 - 包含关键指标生成 ---
def run_snapshot_level_evaluation(model_path):
    """主评估函数 - 快照级别"""
    # 加载数据和构建快照
    handler = get_handler("atlas", False)
    handler.load()
    _, _, _, _, all_snapshots, _, _ = handler.build_graph()

    if not all_snapshots:
        print("错误: 未能构建任何快照。")
        return

    # 获取真实标签
    true_labels = get_true_snapshot_labels(all_snapshots)

    # 生成嵌入并预测
    embedder_class = get_embedder_by_name("prographer")
    embedder = embedder_class(all_snapshots)
    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()

    # 获取RSG嵌入和词汇表
    rsg_embeddings, rsg_vocab = embedder.get_rsg_embeddings()
    print(f"获取RSG嵌入，词汇大小: {len(rsg_vocab)}")

    # 预测异常快照（返回预测标签和差异向量）
    pred_labels, diff_vectors = predict_anomalous_snapshots(snapshot_embeddings, model_path)
    print(f"检测到 {len(diff_vectors)} 个异常快照")

    # 评估指标计算
    eval_start_idx = SEQUENCE_LENGTH_L
    eval_true = true_labels[eval_start_idx:]
    eval_pred = pred_labels[eval_start_idx:]

    tp = np.sum((eval_true == 1) & (eval_pred == 1))
    fp = np.sum((eval_true == 0) & (eval_pred == 1))
    tn = np.sum((eval_true == 0) & (eval_pred == 0))
    fn = np.sum((eval_true == 1) & (eval_pred == 0))

    acc = accuracy_score(eval_true, eval_pred)
    prec = precision_score(eval_true, eval_pred, zero_division=0)
    rec = recall_score(eval_true, eval_pred, zero_division=0)
    f1 = f1_score(eval_true, eval_pred, zero_division=0)

    # 打印评估结果
    print("\n" + "="*50)
    print(" 快照级别评估结果")
    print("="*50)
    print(f" 真阳性 (TP): {tp}")
    print(f" 假阳性 (FP): {fp}")
    print(f" 真阴性 (TN): {tn}")
    print(f" 假阴性 (FN): {fn}")
    print("\n 性能评分:")
    print(f" 准确率: {acc:.4f}")
    print(f" 精确率: {prec:.4f}")
    print(f" 召回率: {rec:.4f}")
    print(f" F1分数: {f1:.4f}")
    print("="*50)

    # 关键指标生成
    generate_key_indicators(all_snapshots, diff_vectors, rsg_embeddings, rsg_vocab)

# 程序入口
if __name__ == '__main__':
    MODEL_PATH = "d:/baseline/process/prographer_detector.pth"
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        sys.exit(1)

    run_snapshot_level_evaluation(MODEL_PATH)