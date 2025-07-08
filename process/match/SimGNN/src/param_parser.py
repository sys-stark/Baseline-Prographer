"""Getting params from the command line."""

import argparse

def parameter_parser():#封装有所有命令行参数定义的方法
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run SimGNN.")#创建参数接收框架    description是描述程序的用途
    #add_argument方法接收参数的种类，参数类型，默认值，参数用途
    parser.add_argument("--training-graphs",
                        nargs="?",
                        default="./dataset/train/",
	                help="Folder with training graph pair jsons.")  # 训练数据仓库地址（默认是仓库A）

    parser.add_argument("--testing-graphs",
                        nargs="?",
                        default="./dataset/test/",
	                help="Folder with testing graph pair jsons.")   # 测试数据仓库地址（默认是仓库B）

    parser.add_argument("--epochs",
                        type=int,
                        default=5,
	                help="Number of training epochs. Default is 5.")    #训练轮数
    #论文中时64、32、16
    parser.add_argument("--filters-1",
                        type=int,
                        default=128,
	                help="Filters (neurons) in 1st convolution. Default is 128.")# 第一层卷积网的"思维通道"数量（默认128条）

    parser.add_argument("--filters-2",
                        type=int,
                        default=64,
	                help="Filters (neurons) in 2nd convolution. Default is 64.")# 第二层通道数（收窄到64条）

    parser.add_argument("--filters-3",
                        type=int,
                        default=32,
	                help="Filters (neurons) in 3rd convolution. Default is 32.")# 第三层通道数（再精简到32条）
    #论文中 NTN layer ， set K to 16
    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
	                help="Neurons in tensor network layer. Default is 16.")# 张量网络的"记忆细胞"数量
    #最终全连接层的中间值
    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
	                help="Bottle neck layer neurons. Default is 16.")# 瓶颈层的"信息漏斗"宽度

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
	                help="Number of graph pairs per batch. Default is 128.")
    #histogram bins
    parser.add_argument("--bins",
                        type=int,
                        default=16,
	                help="Similarity score bins. Default is 16.")# 把相似度分成16个等级来评分

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout probability. Default is 0.5.")#Dropout 层的概率值？？？？

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")# 学习速度（小步快走 vs 大步冒险）

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")# 防止死记硬背的"防沉迷系数"

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")# 是否生成成绩分布图（默认关闭）

    parser.set_defaults(histogram=False)

    parser.add_argument("--save-path",
                        type=str,
                        default=None,
                        help="Where to save the trained model")

    parser.add_argument("--load-path",
                        type=str,   #输入类型是文字
                        default=None,#默认不加载
                        help="Load a pretrained model") #帮助说明

    return parser.parse_args()  # 解析所有参数
