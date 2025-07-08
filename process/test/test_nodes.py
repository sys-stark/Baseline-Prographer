# =================训练=========================
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from process.datahandlers import get_handler
from process.partition import detect_communities

def load_predictions(file_path):
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 确保不是空行
                predictions.append(line)
    return predictions

def test_community_node(communities, ground_truths, predictions):
    y_true = []
    y_pred = []
    tp, fp, tn, fn, total = 0, 0, 0, 0, 0
    for key, community in communities.items():
        for node_name in community:
            pred = int(node_name in predictions)
            label = int(node_name in ground_truths)
            y_true.append(label)
            y_pred.append(pred)
            total +=1
            if label == 1 and pred == 1:
                tp += 1
            elif not label == 1 and pred == 1:
                fp += 1
            elif label == 1 and pred == 0:
                fn += 1
            elif not label == 1 and pred == 0:
                tn += 1
    # === 计算评估指标 ===
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if len(set(y_true)) < 2:
        print("Warning: y_true 只包含一个类别，无法计算 ROC AUC")
        auc = None
    else:
        auc = roc_auc_score(y_true, y_pred)
    fpr = fp / (fp + tn + 1e-10)  # 加上一个小 epsilon 防止除0

    attack_coverage = tp / len(ground_truths)
    workload_reduction = (tp) / (tp + fp + 1e-10)
    print("\n📊评估结果：")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ AUC:       {auc:.4f}")
    print(f"✅ FPR:       {fpr:.4f}")
    print(f"✅ TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"✅ attack_coverage: {attack_coverage}, workload_reduction: {workload_reduction}")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


if __name__ == "__main__":
    data_handler = get_handler("theia", False)
    data_handler.load()
    features, edges, mapp, relations, G = data_handler.build_graph()
    communities = detect_communities(G)
    prediction_path = r"../node_names.txt"
    predictions = load_predictions(prediction_path)
    result = test_community_node(communities, data_handler.all_labels, predictions)
