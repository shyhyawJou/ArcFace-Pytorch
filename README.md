# Overview
Inplementation of a kind of loss function - [ArcFace](https://arxiv.org/abs/1801.07698)

# Usage 
__It is very easy for training and evaluation/inference. 
Just need to alert few code__  

- Replace last fully-connected layer with `ArcFace`
```
from loss import ArcFace
from torchvision import models

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = ArcFace(num_ftrs, num_classes, s=32, m=0.5)
```

- Training phase
```
for img, label in torch_dataloader:
    output = model(img, label)
    loss = loss_func(output, label)
    ......
```

- Evaluation/Inference phase
```
    output = model(img)
```
