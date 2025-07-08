# test_model.py

import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# AnomalyDetector 类的定义需要和 train.py 中的保持一致
# 为了代码独立性，我们在这里重新定义一遍
class AnomalyDetector(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super(AnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, sequence_embeddings):
        lstm_out, _ = self.lstm(sequence_embeddings)
        last_hidden_state = lstm_out[:, -1, :]
        predicted_embedding = self.fc(last_hidden_state)
        return predicted_embedding


def test_model(
        snapshots,
        snapshot_embeddings,
        ground_truth_nodes, # 这是一个包含所有恶意节点名称的集合
        # --- 超参数需要和训练时保持一致 ---
        sequence_length_L=128,
        embedding_dim=256,
        hidden_dim=128,
        num_layers=5,
        dropout_rate=0.2,
        detection_threshold=0.01,
        # --- 模型加载路径 ---
        model_load_path="prographer_detector.pth"
):
    """
    评估 ProGrapher 异常检测模型的性能。

    Args:
        snapshots (list): 完整的图快照列表。
        snapshot_embeddings (np.array): 所有快照的嵌入。
        ground_truth_nodes (set): 包含所有已知恶意节点名称的集合。
        ... (其他超参数)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- 运行模型评估，设备: {device} ---")

    # --- 阶段一：加载已训练的模型 ---
    print("\n--- 阶段一：加载已训练的异常检测器 ---")

    # 检查模型文件是否存在
    if not os.path.exists(model_load_path):
        print(f"错误：找不到已训练的模型文件: {model_load_path}")
        print("请先运行 train_model.py 以训练并保存模型。")
        return

    # 初始化模型结构
    detector_model = AnomalyDetector(embedding_dim, hidden_dim, num_layers, dropout_rate).to(device)

    # 加载已保存的状态字典
    try:
        detector_model.load_state_dict(torch.load(model_load_path, map_location=device))
        print(f"模型状态字典已从 {os.path.abspath(model_load_path)} 加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # --- 阶段二：生成预测和真实标签 ---
    print("\n--- 阶段二：在测试数据上生成预测和真实标签 ---")
    detector_model.eval()  # 将模型切换到评估模式

    y_true = [] # 真实标签列表 (0: 良性, 1: 恶意)
    y_pred = [] # 预测标签列表 (0: 良性, 1: 恶意)

    snapshot_embeddings_tensor = torch.tensor(snapshot_embeddings, dtype=torch.float32)
    criterion = nn.MSELoss()

    if len(snapshot_embeddings_tensor) <= sequence_length_L:
        print("错误：快照数量不足以进行评估。")
        return

    with torch.no_grad():
        # 滑动窗口遍历所有可评估的快照
        for i in range(len(snapshot_embeddings_tensor) - sequence_length_L):
            snapshot_index_to_check = i + sequence_length_L

            # === 获取预测标签 (y_pred) ===
            sequence = snapshot_embeddings_tensor[i : i + sequence_length_L].unsqueeze(0).to(device)
            target = snapshot_embeddings_tensor[snapshot_index_to_check].to(device)
            prediction = detector_model(sequence).squeeze(0)
            reconstruction_error = criterion(prediction, target).item()

            is_pred_malicious = 1 if reconstruction_error > detection_threshold else 0
            y_pred.append(is_pred_malicious)

            # === 获取真实标签 (y_true) ===
            current_snapshot = snapshots[snapshot_index_to_check]
            snapshot_nodes = {v['name'] for v in current_snapshot.vs}

            # 检查当前快照的节点是否与任何一个恶意节点有交集
            is_true_malicious = 1 if not snapshot_nodes.isdisjoint(ground_truth_nodes) else 0
            y_true.append(is_true_malicious)

    if not y_true:
        print("错误：未能生成任何评估标签。")
        return

    # --- 阶段三：计算并打印评估指标 ---
    print("\n--- 阶段三：计算快照级别的评估结果 ---")

    # 计算基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 计算混淆矩阵以获取 TP, FP, FN, TN
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 确保 confusion_matrix 至少返回4个值，以防某个类别完全没有预测
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()


    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"📊 评估指标:")
    print(f"  - 总共评估的快照数: {len(y_true)}")
    print(f"  - 真实恶意快照数: {sum(y_true)}")
    print(f"  - 预测恶意快照数: {sum(y_pred)}")
    print("-" * 20)
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")
    print(f"  - FPR:       {fpr:.4f}")
    print("-" * 20)
    print(f"  - True Positives (TP):  {tp}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - False Negatives (FN): {fn}")
    print(f"  - True Negatives (TN):  {tn}")
    print("\nProGrapher test 完成。")