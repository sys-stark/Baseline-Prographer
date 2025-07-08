"""SimGNN class and runner."""

import glob
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()  #这里的原理
        self.args = args                        #输入的参数
        self.number_labels = number_of_labels   #输入的特征的个数
        self.setup_layers() #层的建立

    #是否要加上直方图数据，不加上的话维度就只考虑NTN
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True: #如果需要直方图的话就是    张量神经网络维度+直方图的维度
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:                           #否则的话就只是张量      神经网络的维度 即蓝色部分
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()    #调用函数计算  节点的维度      ①
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)   #即三个GCN层，GCN是pyg自带的（PyTorch Geometric）
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)  #即三个GCN层
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)  #即三个GCN层
        self.attention = AttentionModule(self.args)            #该函数是    自己定义的。《注意力机制》
        self.tensor_network = TenorNetworkModule(self.args)     #《张量伸进网络》
        #再经过两个线性层（灰色图的那坨）
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)  #输出最总的预测值

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        #前2层GCN都有   激活函数层  和    dropout层，第三层GCN没有？？？？？？？？？？
        #第一层GCN
        features = self.convolution_1(features, edge_index) #GCN
        features = torch.nn.functional.relu(features)   #激活函数层
        features = torch.nn.functional.dropout(features,    #dropout层
                                               p=self.args.dropout,
                                               training=self.training)

        #第二层GCN
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        #第三层GCN
        features = self.convolution_3(features, edge_index)
        return features

    #前向传播不需要显示调用
    def forward(self, data):    #输入的是处理好的数据格式 即 Mytest.txt最后
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"] #边      [1,0,2]说明1与0有双向关系， 2与1只有单项关系，但是无法刻画矢量性？？？？？？？？？？？？？
        edge_index_2 = data["edge_index_2"] #边      [0,1,1]
        features_1 = data["features_1"] #特征
        features_2 = data["features_2"] #特征

        #将one-hot编码用GCN在进行一次编码
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1) #把图1丢尽GCN   
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2) #把图2丢尽GCN   

        #abstract_features_1和abstract_features_2在后面的histogram中会用到，这里先预处理一下（如果有需要的话）？？？？？？？？？？？
        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        #注意力机制层
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        #Neural Tensor  Network模块？？？？？？？？？？？
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score

class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args):#查看如何定义的
        """
        :param args: Arguments object.
        """
        self.global_labels = None
        self.number_of_labels = None
        self.training_graphs = None
        self.testing_graphs = None
        self.args = args    #模型训练类的args变量
        self.initial_label_enumeration()    #最开始初始化训练数据（初步处理）
        self.setup_model()                  #初步处理之后创建模型

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        #第一个类中的不同方法之间，类的属性可以相互通用，这里传入了self.initial_label_enumeration()方法中处理得到的节点特征数量
        self.model = SimGNN(self.args, self.number_of_labels)   #输入的参数   和   特征的个数

    #将raw json data进行       初步处理
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        #按命令行参数批量获取json文件，获取的格式为[0.json，1.json...49.json]
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        #把训练集和测试集合在一起，目的是进行统一性的one - hot(保证覆盖所有元素)
        graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for graph_pair in tqdm(graph_pairs):
            data = process_pair(graph_pair)#读取json
            #lable_1就是该.json中图1节点的属性值，比如[1，2，3，3]去重之后得到的[1，2，3]就是one-hot编码的维度
            #获取one-hot的维度需要遍历完所有的数据，使用set.union方法不断合并
            self.global_labels = self.global_labels.union(set(data["labels_1"]))#用set来去重
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = sorted(self.global_labels)
        #使用enumerate方法给不重复的属性增加索引
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        #这里的label其实不是标签，就是节点的属性，这里      获取节点属性有多少种类型属性          然后传入到136行setup_model（）作为参数
        self.number_of_labels = len(self.global_labels)

    #分批划分数据（输出为python列表格式）
    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)    #将刚获取的JSON随机打乱数据
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph+self.args.batch_size])  #按照设定好的batch-size一个一个加进去
        #返回的batches格式为[[0.json,1.json...49.json],[50.json,51.json...99.json]...]？？？？？？？？？？？？？？？？？？？？？
        return batches

    #将数据转化为pyg需要的那种格式（转置时核心思想）
    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        #该项目场景为有向图
        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]    #自动unpack

        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        # 将输入转化为tarch.Tensor,关键点在于.T转置操作，以适配pyg -------->Mytest.txt中解释
        #T转置操作？？？？？？？？？？？如何转的？？？
        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        #进行one-hot编码并将图对的编码存储在features_1和features_2中，最终的数据格式为[[ont-hot1],[one-hot2]...]
        #？？？？？？？？？？？？？？？？？？？？？？？？？？[ont-hot1]格式为[1,0,0,0],[0,1,0,0,]...
        features_1, features_2 = [], []

        for n in data["labels_1"]:
            features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2

        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

        norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))

        new_data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
        return new_data

    #分batch进行前向传播的， 注意不同的batch之间梯度是不累计的，所以处理完一个batch需要进行梯度清零
    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()  #固定代码
        losses = 0
        #??????????????????????
        for graph_pair in batch:
            # TODO
            data = process_pair(graph_pair) #   统一先将JSON数据转化为python字典，方便进一步处理
            # TODO
            data = self.transfer_to_torch(data)#    关键预处理？？？？？？？？？（将已经one-hot编码的如何变成可以训练的格式）
            target = data["target"] #   上一步处理得到的属性
            prediction = self.model(data)   #   自动调用SimGNN的forward方法
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction)  #   损失函数时MSE
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    #训练入口
    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        #------------------定义一个 优化器  pythrch固定语法------------------
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        # -------------------pythrch固定语法-----
        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")#tqdm库   显示一个进度条
        #-----------------------pythrch固定语法---------------------------
        for epoch in epochs:    #默认轮数是5
            # TODO 加载图
            batches = self.create_batches() #·进行分批计算？？？？？？？？？
            self.loss_sum = 0
            main_index = 0
            #到此 已经打包
            #然后对每个包进行操作
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                #？？？？？？？？？？？？？？？？？？？？
                loss_score = self.process_batch(batch)  #   关键代码（处理每个batch）-----此处就是如何对报的数据进行处理的（如何训练的）
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index #   MSE的实现
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        for graph_pair in tqdm(self.testing_graphs):
            data = process_pair(graph_pair)
            self.ground_truth.append(calculate_normalized_ged(data))
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            self.scores.append(calculate_loss(prediction, target))
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " +str(round(base_error, 5))+".")
        print("\nModel test error: " +str(round(model_error, 5))+".")

    def save(self):
        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))
