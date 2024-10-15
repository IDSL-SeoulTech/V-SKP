from torchvision import models
import torch

resnet50_pretrained = models.vgg16(pretrained=True)

#model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

#print(resnet50_pretrained)