"""SimGNN runner."""

from utils import tab_printer   #将参数以表格形式打印到控制台，方便用户确认配置。
from simgnn import SimGNNTrainer    #模型训练与评估的核心类，负责数据加载、模型构建、训练循环、性能评估及模型保存/加载
from param_parser import parameter_parser   #解析命令行参数（如训练轮数、学习率、模型保存路径等），返回配置对象 args

def main():              #在命令行中敲命令，而非直接执行
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()   #进行命令行参数解析
    tab_printer(args)           #打印参数表格，即可视化命令行参数
    trainer = SimGNNTrainer(args)   #模型训练类接收命令行参数；定义模型（构建模型）；初始化训练器☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆
    if args.load_path:  #如果调用的是训练好的模型
        trainer.load()  #加载已有模型
    else:
        trainer.fit()   #进行模型训练☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆
    # TODO 修改
    trainer.score() #在测试集中评估性能☆☆
    #保存模型
    if args.save_path:  #如果指定保存路径
        trainer.save()  #保存训练后的模型

if __name__ == "__main__":
    main()
