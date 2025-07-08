import copy
import torch
from process.match.dataset import FixedGraphEditDistanceDataset
from process.match.deephunter.graphmatchnet import GraphMatchingScorer
from process.match.evaluation import compute_similarity, auc, compute_metrics
from process.match.deephunter.graphembnet import GraphEncoder, GraphAggregator
from process.match.deephunter.graphmatchnet import GraphMatchingNet

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
            loss='margin',  # other: hamming
            ),
        evaluation=dict(
            batch_size=4),
        seed=8,
    )

def build_datasets(config, communities):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)
    if config['data']['problem'] == 'graph_edit_distance':
        print(f"[数据集构建] 测试社区数: {len(communities)}")
        dataset_params = config['data']['dataset_params']
        dataset_params['dataset_size'] = len(communities)
        validation_set = FixedGraphEditDistanceDataset(**dataset_params, communities=communities)
    else:
        raise ValueError('Unknown problem type: %s' % config['data']['problem'])
    return validation_set

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

    encoder = GraphEncoder(**config['encoder'])
    aggregator = GraphAggregator(**config['aggregator'])
    if config['model_type'] == 'matching':
        model = GraphMatchingNet(
            encoder, aggregator, **config['graph_matching_net'])
    else:
        raise ValueError('Unknown model type: %s' % config['model_type'])
    return model

def test_model(G, communities, node_embeddings, edge_embeddings, model_path="saved_model.pth"):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    config = get_default_config()
    validation_set = build_datasets(config, communities)

    first_data_iter = validation_set.pairs(1, G, node_embeddings, edge_embeddings)
    first_batch_graphs, _ = next(first_data_iter)
    node_feature_dim = first_batch_graphs.node_features.shape[-1]
    edge_feature_dim = first_batch_graphs.edge_features.shape[-1]

    model = build_model(config, node_feature_dim, edge_feature_dim)
    scorer = GraphMatchingScorer(embed_dim=128).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['graph_model'])
    scorer.load_state_dict(checkpoint['scorer'])
    model.to(device)
    scorer.to(device)
    model.eval()
    scorer.eval()

    for batch in validation_set.pairs(config['evaluation']['batch_size'], G, node_embeddings, edge_embeddings):
    # for batch in validation_set.pairs(4, G, node_embeddings, edge_embeddings):
        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
        labels = labels.to(device)

        edge_index = torch.stack([from_idx, to_idx], dim=0).to(device)
        eval_pairs = model(
            x=node_features.to(device),
            edge_index=edge_index,
            batch=None,
            graph_idx=graph_idx.to(device),
            edge_features=edge_features.to(device),
            n_graphs=config['evaluation']['batch_size'] * 2
            # n_graphs = 8
        )
        x, y = reshape_and_split_tensor(eval_pairs, 2)
        similarity = torch.sigmoid(scorer(x, y))
        metrics_result = compute_metrics(similarity, labels)
        print("=== Evaluation Metrics ===")
        print(f"Accuracy:  {metrics_result['Acc']:.4f}")
        print(f"F1 Score:  {metrics_result['F1']:.4f}")
        print(f"AUC:       {metrics_result['AUC']:.4f}")
        print(f"Precision: {metrics_result['Prec']:.4f}")
        print(f"Recall:    {metrics_result['Recall']:.4f}")
        print(f"FPR:       {metrics_result['FPR']:.4f}")


