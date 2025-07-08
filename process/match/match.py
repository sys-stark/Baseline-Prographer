# =============================================================================
#  修改后的 ProGrapher 训练模块 (train_model.py)
#  版本：兼容旧的函数调用，无需修改 train.py
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

# =============================================================================
#  1. 定义 ProGrapher 的异常检测器 (Anomaly Detector)
#     根据论文4.3节，使用 TextRCNN 模型结构
# =============================================================================
class AnomalyDetector(nn.Module):
    """
    使用 TextRCNN 结构实现的异常检测器，以匹配 ProGrapher 论文。
    """
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, kernel_sizes, num_filters):
        super(AnomalyDetector, self).__init__()

        self.lstm = nn.LSTM( # 定义一个 LSTM (长短期记忆网络) 层，这是处理序列数据的核心。
            input_size=embedding_dim, #告诉LSTM，输入序列中每个元素（每个快照嵌入）的向量维度是多少。
            hidden_size=hidden_dim, # LSTM内部“记忆单元”的维度。
            num_layers=num_layers, #叠的LSTM层数。多层网络能学习更复杂的模式。
            batch_first=True,   #规定输入数据的维度顺序为 (批次大小, 序列长度, 特征维度)，这更符合直觉。
            bidirectional=True, #创建了一个双向LSTM。它会同时从前向后和从后向前读取序列。这样做的好处是，在任何一个时间点，模型的输出都同时包含了该点之前和之后的上下文信息，理解更全面
            dropout=dropout if num_layers > 1 else 0 #如果有多层LSTM，就在层与层之间随机丢弃一些连接，防止过拟合。
        )

        self.convs = nn.ModuleList([ #定义一个 ModuleList，它像一个Python列表，但能让PyTorch正确地管理里面的所有层。
            nn.Conv1d(in_channels=hidden_dim * 2, #卷积层的输入通道数
                      out_channels=num_filters, #卷积层输出的通道数
                      kernel_size=k) #卷积核的大小
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout) #定义一个 Dropout 层
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embedding_dim)  #定义一个 全连接（线性）层 (nn.Linear)

    def forward(self, sequence_embeddings):
        lstm_out, _ = self.lstm(sequence_embeddings) #首先通过双向LSTM层。
        conv_in = lstm_out.permute(0, 2, 1) #permute 用于交换张量的维度。

        # 优化后的卷积和池化操作
        pooled_outputs = []
        for conv in self.convs: #遍历我们创建的每一个卷积层
            conv_out = F.relu(conv(conv_in))  #将 conv_in 输入到卷积层，然后用 F.relu 激活函数处理。
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)  #最大池化
            pooled_outputs.append(pooled)  #将每个卷积核提取出的最强特征信号

        concatenated = torch.cat(pooled_outputs, dim=1)  #将所有来自不同尺寸卷积核的池化结果（特征向量）拼接成一个更长的向量。
        dropped_out = self.dropout(concatenated)   #将拼接后的特征向量通过 Dropout 层。
        predicted_embedding = self.fc(dropped_out) #将处理后的特征向量输入到最后的全连接层，得到最终的预测结果——一个维度为 embedding_dim 的向量。

        return predicted_embedding

# =============================================================================
#  2. 训练模型的主函数 (兼容旧的调用签名)
# =============================================================================
def train_model(
        snapshots,          # 函数内部不再使用
        snapshot_embeddings,
        rsg_embeddings,     # 函数内部不再使用
        rsg_vocab,          # 函数内部不再使用
        # --- 使用论文中定义的超参数，并加入 TextRCNN 特有的参数 ---
        sequence_length_L=12, #序列长度
        embedding_dim=256, #嵌入维度
        hidden_dim=128, # 隐藏层维度
        num_layers=5,  #LSTM层数
        dropout_rate=0.2, #Dropout比率
        # --- 新增的 TextRCNN 参数，提供默认值 ---
        kernel_sizes=[3, 4, 5],  #卷积核尺寸
        num_filters=100,  #滤波器数量
        # --- 其他训练超参数 ---
        seq_lr=3e-4,  #学习率
        seq_epochs=30, #训练周期数
        batch_size=32, #批处理大小
        # --- 模型保存路径参数 ---
        model_save_path="d:/baseline/process/prographer_detector.pth" # 使用一个明确的默认路径
):
    """
    实现了 ProGrapher 的异常检测器训练 (TextRCNN 版本)。
    这个函数签名兼容旧的调用方式，无需修改调用方代码。

    Args:
        snapshots, rsg_embeddings, rsg_vocab: 为了兼容性而保留，但不再使用。
        snapshot_embeddings (np.array): 编码器输出的快照嵌入。
        ... (其他超参数)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- ProGrapher 训练后端运行在: {device} ---")
    print("\n--- 阶段一：训练序列异常检测器 (TextRCNN) ---")

    # 1. 准备数据
    snapshot_embeddings_tensor = torch.tensor(snapshot_embeddings, dtype=torch.float32) #转为张量
    if len(snapshot_embeddings_tensor) <= sequence_length_L:  #数据处理
        print(f"错误：快照数量 ({len(snapshot_embeddings_tensor)}) 不足以构成一个长度为 {sequence_length_L} 的序列。")
        return

    sequences, targets = [], []  #sequences存放模型的输入，tragets存放真实标签
    for i in range(len(snapshot_embeddings_tensor) - sequence_length_L):  #滑动窗口
        sequences.append(snapshot_embeddings_tensor[i : i + sequence_length_L]) #取出长度为L的片段作为输入序列
        targets.append(snapshot_embeddings_tensor[i + sequence_length_L]) #获得真实序列

    if not sequences:
        print("错误：无法创建任何训练序列。")
        return

    print(f"成功创建 {len(sequences)} 个训练序列。")

    dataset = torch.utils.data.TensorDataset(torch.stack(sequences), torch.stack(targets))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型、优化器和损失函数
    detector_model = AnomalyDetector(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout_rate,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters
    ).to(device)  #创建一个AnomalyDetector

    optimizer = optim.Adam(detector_model.parameters(), lr=seq_lr) #创建Adam优化器
    criterion = nn.MSELoss()  #MSEloss，比较真实与预测之间的差异

    # 3. 训练循环
    detector_model.train()
    for epoch in range(seq_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{seq_epochs}", leave=True)
        for seq_batch, target_batch in progress_bar: #从 dataloader 中一次取出一批（batch）数据
            seq_batch, target_batch = seq_batch.to(device), target_batch.to(device) # 将这一批数据也移动到指定的设备（CPU或GPU）

            optimizer.zero_grad()  #在计算新的梯度之前，必须清除上一步留下的旧梯度。
            predicted_batch = detector_model(seq_batch) #将一批输入序列喂给模型，得到一批预测结果。
            loss = criterion(predicted_batch, target_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader) if dataloader else 0
        print(f"Epoch {epoch+1}/{seq_epochs}, Avg. MSE Loss: {avg_loss:.6f}")

    print("异常检测器训练完成。")

    # 4. 保存模型
    if model_save_path:
        try:
            save_dir = os.path.dirname(model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(detector_model.state_dict(), model_save_path)
            print(f"模型状态字典已成功保存至: {os.path.abspath(model_save_path)}")
        except Exception as e:
            print(f"保存模型失败: {e}")