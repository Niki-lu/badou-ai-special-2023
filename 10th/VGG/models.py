import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, num_classes) -> None:
        super(VGG,self).__init__()
        self.features=nn.Sequential(
            self.convblock(3,64),
            self.convblock(64,64),
            nn.MaxPool2d(kernel_size=2,stride=2),
            self.convblock(64,128),
            self.convblock(128,128),
            nn.MaxPool2d(kernel_size=2,stride=2),
            self.convblock(128,256),
            self.convblock(256,256),
            self.convblock(256,256),
            self.convblock(256,256),
            nn.MaxPool2d(kernel_size=2,stride=2),
            self.convblock(256,512),
            self.convblock(512,512),
            self.convblock(512,512),
            self.convblock(512,512),
            nn.MaxPool2d(kernel_size=2,stride=2),
            self.convblock(512,512),
            self.convblock(512,512),
            self.convblock(512,512),
            self.convblock(512,512),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.avgpool=nn.AdaptiveAvgPool2d((7,7))
        self.classifier=nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
        )
    def convblock(self,input_channels,output_channels,kernel_size=3,stride=1,padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x
    
vgg19=VGG(100)
print(vgg19)