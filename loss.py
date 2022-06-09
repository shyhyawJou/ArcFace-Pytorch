import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.linalg import norm as torch_norm

"""
class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=64, m=0.5):
        super().__init__()
        self.m = m
        self.s = s
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None):
        if label is None:
            logit = self.fc(x)
        else:
            # w_L2 shape (cout, 1), x_L2 shape (b, 1)
            # w_L2 or x_L2 need to be transposed
            w_L2 = torch.sum(self.fc.weight.detach() ** 2, 1, True).view(1, -1)
            x_L2 = torch.sum(x ** 2, 1, True)

            logit = self.fc(x)
            cos = logit / torch.pow(w_L2 * x_L2, 0.5)
            theta = torch.arccos(cos)

            one_hot = F.one_hot(label, num_classes=self.cout)
            logit = one_hot * torch.cos(theta + self.m) + (1 - one_hot) * logit
            logit *= self.s
        
        return logit
"""


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
    
 

    
def mixup_loss(lf, output, label1, label2, factor):
    loss = factor * lf(output, label1) + (1 - factor) * lf(output, label2)
    return loss


m = ArcFace(2, 2)

x = torch.randn(3, 2)
label = torch.tensor([0, 1, 1])
label = None

loss = m(x, label)
print(loss)