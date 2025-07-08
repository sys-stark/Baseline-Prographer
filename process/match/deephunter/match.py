import collections
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from process.match.dataset import GraphEditDistanceDataset, FixedGraphEditDistanceDataset
from process.match.deephunter.graphembnet import GraphEncoder, GraphAggregator
from process.match.deephunter.graphmatchnet import GraphMatchingNet, GraphMatchingScorer
from process.match.evaluation import compute_similarity, auc
from process.match.loss import pairwise_loss


def split_communities_renumbered(communities, ratio=0.9):
    """按比例拆分并重新编号 community id"""
    sorted_ids = sorted(communities.keys())
    split_idx = int(len(sorted_ids) * ratio)

    train_ids = sorted_ids[:split_idx]
    eval_ids = sorted_ids[split_idx:]

    communities_train = {i: communities[cid] for i, cid in enumerate(train_ids)}
    communities_eval = {i: communities[cid] for i, cid in enumerate(eval_ids)}

    return communities_train, communities_eval


def build_datasets(config, communities):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)
    communities_train, communities_eval = split_communities_renumbered(communities)
    if config['data']['problem'] == 'graph_edit_distance':
        print(f"[数据集构建] 训练社区数: {len(communities_train)}, 验证社区数: {len(communities_eval)}")

        dataset_params = config['data']['dataset_params']
        training_set = GraphEditDistanceDataset(**dataset_params, communities=communities_train)

        dataset_params['dataset_size'] = len(communities_eval)
        validation_set = FixedGraphEditDistanceDataset(**dataset_params, communities=communities_eval)
    else:
        raise ValueError('Unknown problem type: %s' % config['data']['problem'])
    return training_set, validation_set


def get_default_config():
    """The default configs."""
    model_type = 'matching'
    # Set to `embedding` to use the graph embedding net.
    node_state_dim = 32
    edge_state_dim = 16
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        node_update_type='gru',
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type=model_type)
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'dotproduct'  # other: euclidean, cosine
    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=1,
            edge_hidden_sizes=[edge_state_dim]),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type='sum'),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        data=dict(
            problem='graph_edit_distance',
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=0.1,
                n_changes_negative=0.5)),
        training=dict(
            batch_size=20,
            learning_rate=1e-4,
            mode='pair',
            loss='margin',  # other: hamming
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            # n_training_steps=500000,
            n_training_steps=500,
            # Print training information every this many training steps.
            # print_after=100,
            print_after=10,
            # Evaluate on validation set every `eval_after * print_after` steps.
            eval_after=1,
            # eval_after=10
            ),
        evaluation=dict(
            batch_size=20),
        seed=8,
    )

def build_model(config, node_feature_dim, edge_feature_dim):
    """Create model for training and evaluation.

    Args:
      config: a dictionary of configs, like the one created by the
        `get_default_config` function.
      node_feature_dim: int, dimensionality of node features.
      edge_feature_dim: int, dimensionality of edge features.

    Returns:
      tensors: a (potentially nested) name => tensor dict.
      placeholders: a (potentially nested) name => tensor dict.
      AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

    Raises:
      ValueError: if the specified model or training settings are not supported.
    """
    config['encoder']['node_feature_dim'] = node_feature_dim
    config['encoder']['edge_feature_dim'] = edge_feature_dim

    # TODO 修改GCN
    encoder = GraphEncoder(**config['encoder'])
    aggregator = GraphAggregator(**config['aggregator'])
    if config['model_type'] == 'matching':
        model = GraphMatchingNet(
            encoder, aggregator, **config['graph_matching_net'])
    else:
        raise ValueError('Unknown model type: %s' % config['model_type'])

    optimizer = torch.optim.Adam((model.parameters()),
                                 lr=config['training']['learning_rate'], weight_decay=1e-5)

    return model, optimizer

def get_graph(batch):
    if len(batch) != 2:
        # if isinstance(batch, GraphData):
        graph = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        return node_features, edge_features, from_idx, to_idx, graph_idx
    else:
        graph, labels = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        labels = torch.from_numpy(labels).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx, labels

def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split



def train_model(G, communities, node_embeddings, edge_embeddings):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # 加载数据
    config = get_default_config()
    training_set, validation_set = build_datasets(config, communities)

    training_data_iter = training_set._pairs(config['training']['batch_size'], G, node_embeddings, edge_embeddings)
    first_batch_graphs, _ = next(training_data_iter)

    # 初始化模型
    node_feature_dim = first_batch_graphs.node_features.shape[-1]
    edge_feature_dim = first_batch_graphs.edge_features.shape[-1]
    # 模型得改下
    model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
    model.to(device)

    scorer = GraphMatchingScorer(embed_dim=128).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 创建存储训练过程的指标
    accumulated_metrics = collections.defaultdict(list)
    # 计算每个 batch 里的图数量
    training_n_graphs_in_batch = config['training']['batch_size']
    if config['training']['mode'] == 'pair':
        training_n_graphs_in_batch *= 2
    else:
        raise ValueError('Unknown training mode: %s' % config['training']['mode'])

    # 训练循环
    t_start = time.time()
    for i_iter in range(config['training']['n_training_steps']):
        model.train(mode=True)
        # 解析 batch 数据
        batch = next(training_data_iter)
        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
        labels = labels.to(device)

        #  前向传播
        edge_index = torch.stack([from_idx, to_idx], dim=0).to(device)
        graph_vectors = model(
            node_features.to(device),
            edge_index,
            None,
            graph_idx.to(device),
            edge_features=edge_features.to(device),
            n_graphs = training_n_graphs_in_batch
        )
        #  计算损失
        x, y = reshape_and_split_tensor(graph_vectors, 2)

        is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
        is_neg = 1 - is_pos
        n_pos = torch.sum(is_pos)
        n_neg = torch.sum(is_neg)
        sim = scorer(x, y)
        loss = loss_fn(sim, labels.float())
        sim_sigmoid = torch.sigmoid(sim)
        sim_pos = torch.sum(sim_sigmoid * is_pos) / (n_pos + 1e-8)
        sim_neg = torch.sum(sim_sigmoid * is_neg) / (n_neg + 1e-8)

        graph_vec_scale = torch.mean(graph_vectors ** 2)
        if config['training']['graph_vec_regularizer_weight'] > 0:
            loss = loss.add(config['training']['graph_vec_regularizer_weight'] *
                            0.5 * graph_vec_scale)

        # 反向传播 & 更新参数
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))  #
        nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
        optimizer.step()

        sim_diff = sim_pos - sim_neg
        accumulated_metrics['loss'].append(loss)
        accumulated_metrics['sim_pos'].append(sim_pos)
        accumulated_metrics['sim_neg'].append(sim_neg)
        accumulated_metrics['sim_diff'].append(sim_diff)

        # evaluation
        if (i_iter + 1) % config['training']['print_after'] == 0:
            # 打印训练参数
            metrics_to_print = {
                k: torch.mean(v[0]) for k, v in accumulated_metrics.items()}
            info_str = ', '.join(
                ['%s %.4f' % (k, v) for k, v in metrics_to_print.items()])
            # reset the metrics
            accumulated_metrics = collections.defaultdict(list)

            # 计算AUC / Triplet Accuracy
            # 评估
            if ((i_iter + 1) // config['training']['print_after'] %
                    config['training']['eval_after'] == 0):
                model.eval()
                with torch.no_grad():
                    accumulated_pair_auc = []
                    for batch in validation_set.pairs(config['evaluation']['batch_size'], G, node_embeddings, edge_embeddings):
                        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                        labels = labels.to(device)
                        edge_index = torch.stack([from_idx, to_idx], dim=0).to(device)
                        eval_pairs = model(
                            x=node_features.to(device),
                            edge_index=edge_index.to(device),
                            batch = None,
                            graph_idx=graph_idx.to(device),
                            edge_features=edge_features.to(device),
                            n_graphs=config['evaluation']['batch_size'] * 2  # 传递 n_graphs
                        )
                        x, y = reshape_and_split_tensor(eval_pairs, 2)
                        similarity = torch.sigmoid(scorer(x, y))
                        pair_auc = auc(similarity, labels)
                        accumulated_pair_auc.append(pair_auc)

                    eval_metrics = {
                        'pair_auc': np.mean(accumulated_pair_auc), }
                    info_str += ', ' + ', '.join(
                        ['%s %.4f' % ('val/' + k, v) for k, v in eval_metrics.items()])
                model.train()
            print('iter %d, %s, time %.2fs' % (
                i_iter + 1, info_str, time.time() - t_start))
            t_start = time.time()
    save_path = "saved_model.pth"
    torch.save({
        'graph_model': model.state_dict(),
        'scorer': scorer.state_dict()
    }, save_path)
    print(f"模型已保存到 {save_path}")
    print("训练结束")
