import os
import torch
import torch.nn as nn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .Utills import iou

class CNN(nn.Module): 
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_batch_norm=True,
                 **kwargs
                 ):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm 
        
    def forward(self, x): 
        x = self.conv(x)
        if self.use_batch_norm: 
            x = self.bn(x)
            return self.activation(x)
        else: 
            return x
    
class Residual(nn.Module): 
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__() 
        res_layers = [] 
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size = 1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1),
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        
    def forward(self, x):
        for layer in self.layers: 
            residual = x 
            x = layer(x)
            if self.use_residual: 
                x = x + residual 
        return x
    
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2*in_channels, (num_classes + 5)  * 3, kernel_size=1),
        )
        self.num_classes = num_classes
        
    def forward(self, x): 
        output = self.pred(x)
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0, 1, 3, 4, 2)
        return output
    
    
    
    
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.layers = nn.ModuleList([
            CNN(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNN(32, 64, kernel_size=3, stride=2, padding=1),
            Residual(64, num_repeats=1),
            CNN(64, 128, kernel_size=3, stride=2, padding=1),
            Residual(128, num_repeats=2),
            CNN(128, 256, kernel_size=3, stride=2, padding=1),
            Residual(256, num_repeats=8),
            CNN(256, 512, kernel_size=3, stride=2, padding=1),
            Residual(512, num_repeats=8),
            CNN(512, 1024, kernel_size=3, stride=2, padding=1),
            Residual(1024, num_repeats=4),
            CNN(1024, 512, kernel_size=1, stride=1, padding=0),
            CNN(512, 1024, kernel_size=3, stride=1, padding=1),
            Residual(1024, use_residual=False, num_repeats=1),
            CNN(1024, 512, kernel_size=1, stride=1, padding=0),
            ScalePrediction(512, num_classes=num_classes),
            CNN(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNN(768, 256, kernel_size=1, stride=1, padding=0),
            CNN(256, 512, kernel_size=3, stride=1, padding=1),
            Residual(512, use_residual=False, num_repeats=1),
            CNN(512, 256, kernel_size=1, stride=1, padding=0),
            ScalePrediction(256, num_classes=num_classes),
            CNN(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNN(384, 128, kernel_size=1, stride=1, padding=0),
            CNN(128, 256, kernel_size=3, stride=1, padding=1),
            Residual(256, use_residual=False, num_repeats=1),
            CNN(256, 128, kernel_size=1, stride=1, padding=0),
            ScalePrediction(128, num_classes=num_classes)
        ])
        
    def forward(self, x):
        outputs = []
        route_connections = [] 
        for layer in self.layers: 
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            
            if isinstance(layer, Residual) and layer.num_repeats == 8: 
                route_connections.append(x)
                
            elif isinstance(layer, nn.Upsample): 
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
                
        return outputs
        
class YOLO_LOSS(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        obj = target[..., 0] == 1 
        no_obj = target[..., 0] == 0
        
        no_object_loss = self.bce(
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]),
        )
        
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors], dim=-1)
        
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        
        object_loss = self.mse(self.sigmoid(pred[..., :1][obj]), ious * target[..., 0:1][obj])
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])
        class_loss = self.cross_entropy((pred[..., 5:][obj]), target[..., 5][obj].long())
        
        return (box_loss + object_loss + no_object_loss + class_loss)
                
        
# Testing YOLO v3 model
if __name__ == "__main__":
    # Setting number of classes and image size
    num_classes = 20
    IMAGE_SIZE = 416

    # Creating model and testing output shapes
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

    # Asserting output shapes
    assert model(x)[0].shape == (1, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (1, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (1, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Output shapes are correct!")

    