# =================图神经网络=========================
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
from gensim.models import Word2Vec
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec

class PositionalEncoder:

    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]


class EpochSaver(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        model.save('word2vec_theia_E3.model')
        self.epoch += 1

class EpochLogger(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def infer(document, path):
    encoder = PositionalEncoder(30)
    w2vmodel = Word2Vec.load(path)
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]

    if not word_embeddings:
        return np.zeros(30)

    output_embedding = torch.tensor(np.array(word_embeddings), dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)