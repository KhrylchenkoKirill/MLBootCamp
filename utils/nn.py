from torch import nn
import torch

class LinearModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.cls = nn.Linear(n_features, 4)
        
    def forward(self, x):
        return self.cls(x)
    
    def initialize(self, features, regularize=False):
        self.regularize = regularize
        if self.regularize:
            self.reg_positions = []
        n = len(features)
        weight = []
        for idx, word in enumerate(['Xmin_min', 'Ymin_min', 'Xbias_opt', 'Ybias_opt']):
            pos = features.index(word)
            if self.regularize:
                self.reg_positions.append(n * idx + pos)
            weight.append([0] * pos + [1] + [0] * (n - pos - 1))
        self.cls.weight.data = torch.tensor(weight).float()
        self.cls.bias.data = torch.zeros(4).float()
        
    def L1_loss(self, coef=1.):
        flattened_weights = self.cls.weight.flatten()
        
        term_1 = coef * (1. - flattened_weights[self.reg_positions]).abs().sum()
        
        term_2 = flattened_weights[:self.reg_positions[0]].abs().sum()
        for i in range(len(self.reg_positions) - 1):
            term_2 += flattened_weights[self.reg_positions[i] + 1 : self.reg_positions[i + 1]].abs().sum()
        term_2 += flattened_weights[self.reg_positions[i + 1] + 1: ].abs().sum() 
        term_2 += self.cls.bias.flatten().abs().sum()
        term_2 *= coef 
        return term_1 + term_2
    
    def L2_loss(self, coef=1.):
        flattened_weights = self.cls.weight.flatten()
        
        term_1 = coef * (1. - flattened_weights[self.reg_positions]).pow(2).sum()
        
        term_2 = flattened_weights[:self.reg_positions[0]].pow(2).sum()
        for i in range(len(self.reg_positions) - 1):
            term_2 += flattened_weights[self.reg_positions[i] + 1 : self.reg_positions[i + 1]].pow(2).sum()
        term_2 += flattened_weights[self.reg_positions[i + 1] + 1: ].pow(2).sum()
        term_2 += self.cls.bias.flatten().pow(2).sum()
        term_2 *= coef / 2.
        return term_1 + term_2