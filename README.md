# Overview
Inplementation of a kind of loss function - [ArcFace](https://arxiv.org/abs/1801.07698)

# Usage 
__It is very easy for evaluation/inference.__  
__However, you need to alert few code for training__, 

- add `label` argument to model's forward function
  ```
  class Your_Model(nn.Module):
        def __init__(self):
            self.conv1 = nn.Conv2d(...)
            ..............
            self.classifier = nn.Linear(...)
        
        def forward(self, x, label=None):
            x = self.conv1(x)
            ...................
            if label is None:
                output = self.classifier(x)
            else:
                output = self.classifier(x, label)
            return output
  ```

- Replace last fully-connected layer of your model with `ArcFace`
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

# Note
If your training is hard to converge, you can set m to smaller.
