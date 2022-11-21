# Overview
[ArcFace](https://arxiv.org/abs/1801.07698)

# Usage  
You need to alert your model. 

  - add `label` argument to model's forward function
  - add `label` as classifier's input
    ```
    class Your_Model(nn.Module):
          def __init__(self):
              ..............
              self.classifier = ....

          def forward(self, x, label=None):
              ...................
              output = self.classifier(x, label)
              return output
    ```


- __Training phase__
1. Replace last fully-connected layer of your model with `ArcFace`  
   Take ResNet18 for example.
```
from loss import ArcFace
from torchvision import models

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = ArcFace(num_ftrs, num_classes, m=0.5)
```
2. in the training loop, add `label` as model's input argument  
   (Generally, loss_func is `nn.CrossEntropyLoss()`)
```
for img, label in torch_dataloader:
    output = model(img, label)
    loss = loss_func(output, label)
    ......
```

- __Evaluation/Inference phase__  
Don't need to alert anything

# Note
- If your training is hard to converge, you can set m to smaller.
