import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.linalg import norm as torch_norm



class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=32, m=0.5):
        super().__init__()
        self.m = m
        self.s = s
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)                              

    def forward(self, x, label=None):
        if label is None:
            w_L2 = torch_norm(self.fc.weight.detach(), dim=1, keepdim=True)
            x_L2 = torch_norm(x, dim=1, keepdim=True)
            logit = F.linear(x / x_L2, self.fc.weight / w_L2)
        else:
            one_hot = F.one_hot(label, num_classes=self.cout)
            w_L2 = torch_norm(self.fc.weight.detach(), dim=1, keepdim=True)
            x_L2 = torch_norm(x, dim=1, keepdim=True)
            cos = F.linear(x / x_L2, self.fc.weight / w_L2)
            theta_yi = torch.acos(cos * one_hot)
            logit = torch.cos(theta_yi + self.m) * one_hot + cos * (1 - one_hot)
            logit = logit * self.s

        return logit
