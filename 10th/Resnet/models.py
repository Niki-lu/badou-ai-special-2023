import torch
import torch.nn as nn

class Resnet50(nn.Module):
    def __init__(self, num_classes) -> None:
        super(Resnet50,self).__init__()
        self.num_classes=num_classes
        self.head=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False)
        )

    def convBlock(self,x,filters,kernel_size=3,stride=2):
        long=nn.Sequential(
                nn.Conv2d(filters[0][0],filters[0][1],kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(filters[0][1],eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                nn.Conv2d(filters[1][0],filters[1][1],kernel_size=kernel_size,stride=stride,padding=1,bias=False),
                nn.BatchNorm2d(filters[1][1],eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                nn.Conv2d(filters[2][0],filters[2][1],kernel_size=1,bias=False),
                nn.BatchNorm2d(filters[2][1],eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                nn.ReLU(inplace=True))
        downsample=nn.Sequential(
            nn.Conv2d(filters[0][0],filters[2][1],kernel_size=kernel_size,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(filters[2][1],eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
        )
        x=long(x)+downsample(x)
        return x
    def identityBlock(self,x,filters,kernel_size=3,stride=1):
        long=nn.Sequential(
                nn.Conv2d(filters[0][0],filters[0][1],kernel_size=1,bias=False),
                nn.BatchNorm2d(filters[0][1],eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                nn.Conv2d(filters[1][0],filters[1][1],kernel_size=kernel_size,stride=stride,padding=1,bias=False),
                nn.BatchNorm2d(filters[1][1],eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                nn.Conv2d(filters[2][0],filters[2][1],kernel_size=1,bias=False),
                nn.BatchNorm2d(filters[2][1],eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                nn.ReLU(inplace=True))
        x=x+long(x)
        return x
    
    def forward(self,x):
        x=self.head(x)
        x=self.convBlock(x,[[64,64],[64,64],[64,256]])
        x=self.identityBlock(x,[[256,64],[64,64],[64,256]])
        x=self.identityBlock(x,[[256,64],[64,64],[64,256]])
        x=self.convBlock(x,[[256,128],[128,128],[128,512]])
        x=self.identityBlock(x,[[512,128],[128,128],[128,512]])
        x=self.identityBlock(x,[[512,128],[128,128],[128,512]])
        x=self.identityBlock(x,[[512,128],[128,128],[128,512]])
        x=self.convBlock(x,[[512,256],[256,256],[256,1024]])
        x=self.identityBlock(x,[[1024,256],[256,256],[256,1024]])
        x=self.identityBlock(x,[[1024,256],[256,256],[256,1024]])
        x=self.identityBlock(x,[[1024,256],[256,256],[256,1024]])
        x=self.identityBlock(x,[[1024,256],[256,256],[256,1024]])
        x=self.identityBlock(x,[[1024,256],[256,256],[256,1024]])
        x=self.convBlock(x,[[1024,512],[512,512],[512,2048]])
        x=self.identityBlock(x,[[2048,512],[512,512],[512,2048]])
        x=self.identityBlock(x,[[2048,512],[512,512],[512,2048]])
        x=nn.AdaptiveAvgPool2d((1,1))(x)
        x=x.view(x.size(0),-1)
        x=nn.Linear(in_features=2048,out_features=self.num_classes)(x)
        return x