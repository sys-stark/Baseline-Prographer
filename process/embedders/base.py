from abc import ABC, abstractmethod

class GraphEmbedderBase(ABC):
    def __init__(self, G, features, mapp):
        self.G = G
        self.features = features
        self.mapp = mapp

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def embed_nodes(self):
        pass

    @abstractmethod
    def embed_edges(self):
        pass