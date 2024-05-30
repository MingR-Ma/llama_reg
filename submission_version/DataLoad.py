from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
from random import shuffle
import json
import itertools


class DataGenerator(Dataset):
    def __init__(self, train_names,train_path, mode='train'):
        """

        :param json_file:
        :param h5_file:
        """
        self.mode = mode
        self.train_path=train_path

        self.pair = list(itertools.permutations(train_names, 2))
        shuffle(self.pair)

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, index):

        self.data_A=np.load(self.train_path+self.pair[index][0]+'.npy')
        self.data_B=np.load(self.train_path+self.pair[index][1]+'.npy')

        return torch.Tensor(self.data_A).unsqueeze(0),torch.Tensor(self.data_B).unsqueeze(0)

