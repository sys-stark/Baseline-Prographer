from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    def __init__(self, base_path, train):
        self.base_path = base_path
        self.train = train
        self.all_dfs = []                 # 多个数据帧
        self.all_netobj2pro = {}  # 网络对象 UUID → 属性字符串
        self.all_subject2pro = {}  # 进程 UUID → 属性字符串
        self.all_file2pro = {}  # 文件 UUID → 属性字符串
        self.use_df = []
        self.all_labels = []
        self.total_loaded_bytes = 0

    @abstractmethod
    def load(self):
        """加载原始数据，返回预处理好的 DataFrame 以及属性映射字典"""
        pass

    @abstractmethod
    def build_graph(self):
        """构建图及其语料"""
        pass
