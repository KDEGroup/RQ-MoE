import numpy as np
from faiss.contrib.datasets import Dataset
from faiss.contrib.datasets import dataset_from_name as dataset_from_name_faiss
from faiss.contrib.vecs_io import fvecs_read, ivecs_read


class DatasetFB_ssnpp(Dataset):
    def __init__(self, nb_M=1):
        Dataset.__init__(self)
        assert nb_M == 1
        self.d, self.nt, self.nb, self.nq = 256, int(1e7), int(nb_M * 10**6), 10000
        self.basedir = "data/fb_ssnpp/"

    def get_queries(self):
        return np.load(self.basedir + "queries.npy")

    def get_train(self, maxtrain=None):
        if maxtrain is None:
            maxtrain = self.nt
        xt = np.load(self.basedir + "training_set10010k.npy", mmap_mode="r")
        if maxtrain <= 10010000:
            return np.array(xt[:maxtrain])
        else:
            raise NotImplementedError

    def get_database(self):
        return np.load(self.basedir + "database1M.npy")

    def get_groundtruth(self, k=None):
        gt = np.load(self.basedir + "ground_truth1M.npy")
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt


class DatasetContrieverEmb(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.d, self.nt, self.nb, self.nq = 768, int(20e6), int(1e6), 10000
        self.basedir = "data/contriever/"

    def get_queries(self):
        return np.load(self.basedir + "queries.npy").astype("float32")

    def get_train(self, maxtrain=None):
        if maxtrain is None:
            maxtrain = self.nt
        xt = np.load(self.basedir + "training_set.npy", mmap_mode="r")
        if maxtrain <= self.nt:
            return np.array(xt[:maxtrain]).astype("float32")
        else:
            raise NotImplementedError

    def get_database(self):
        return np.load(self.basedir + "database1M.npy").astype("float32")

    def get_groundtruth(self, k=None):
        gt = np.load(self.basedir + "ground_truth1M.npy")
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt


available_names = [
    "bigann1M",
    "bigann10M",
    "bigann100M",
    "bigann1B",
    "deep1M",
    "deep10M",
    "deep100M",
    "deep1B",
    "FB_ssnpp1M",
    "Contriever1M",
]

def dataset_from_name(name):
    if name == "FB_ssnpp1M":
        return DatasetFB_ssnpp()
    elif name == "Contriever1M":
        return DatasetContrieverEmb()
    else:
        return dataset_from_name_faiss(name)

