import torch
import torch.nn as nn

class Conv(nn.Module):
    default_act=nn.ReLU()
    def __init__(self,c1,c2,k,s=1,p=0,act=True):
        super().__init__()
        self.conv=nn.Conv2d(c1,c2,kernel_size=k,stride=s,padding=p)
        self.bn=nn.BatchNorm2d(c2)
        self.act=self.default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()
    
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class InceptionBlock1(nn.Module):
    def __init__(self,c_in,out1,hid2,out2,hid3,out3,out4):
        super().__init__()

        self.branch1=Conv(c_in,out1,k=1,s=1,p=0)
        self.branch2=nn.Sequential(
            Conv(c_in,hid2,k=1,s=1,p=0),
            Conv(hid2,out2,k=3,s=1,p=1)
        )
        self.branch3=nn.Sequential(
            Conv(c_in,hid3,k=1,s=1,p=0),
            Conv(hid3,out3,k=3,s=1,p=1),
            Conv(out3,out3,k=3,s=1,p=1)
        )

        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            Conv(c_in,out4,k=1,s=1,p=0)
        )

    def forward(self,x):
        return torch.cat((self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)),1)
    
class InceptionBlock2_1(nn.Module):
    def __init__(self,c_in,out2,hid3,out3,stride):
        super().__init__()

        self.branch2=Conv(c_in,out2,k=3,s=stride,p=0)
        self.branch3=nn.Sequential(
            Conv(c_in,hid3,k=1,s=1,p=0),
            Conv(hid3,out3,k=3,s=1,p=1),
            Conv(out3,out3,k=3,s=stride,p=0)
        )

        self.branch4=nn.MaxPool2d(kernel_size=3,stride=stride,padding=0)
            

    def forward(self,x):
        return torch.cat((self.branch2(x),self.branch3(x),self.branch4(x)),1)
    
    
class InceptionBlock2_2(nn.Module):
    def __init__(self,c_in,out1,hid2,out2,hid3,out3,out4):
        super().__init__()

        self.branch1=Conv(c_in,out1,k=1,s=1,p=0)
        self.branch2=nn.Sequential(
            Conv(c_in,hid2,k=1,s=1,p=0),
            Conv(hid2,hid2,k=(1,7),s=1,p=(0,3)),
            Conv(hid2,out2,k=(7,1),s=1,p=(3,0))
        )
        self.branch3=nn.Sequential(
            Conv(c_in,hid3,k=1,s=1,p=0),
            Conv(hid3,hid3,k=(7,1),s=1,p=(3,0)),
            Conv(hid3,hid3,k=(1,7),s=1,p=(0,3)),
            Conv(hid3,hid3,k=(7,1),s=1,p=(3,0)),
            Conv(hid3,out3,k=(1,7),s=1,p=(0,3))
        )

        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            Conv(c_in,out4,k=1,s=1,p=0)
        )
    def forward(self,x):
        return torch.cat((self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)),1)   
    

class InceptionBlock3_1(nn.Module):
    def __init__(self,c_in,hid2,out2,hid3,out3):
        super().__init__()
        self.branch2=nn.Sequential(
            Conv(c_in,hid2,k=1,s=1),
            Conv(hid2,out2,k=3,s=2,p=0))
        self.branch3=nn.Sequential(
            Conv(c_in,hid3,k=1,s=1),
            Conv(hid3,hid3,k=(1,7),s=1,p=(0,3)),
            Conv(hid3,hid3,k=(7,1),s=1,p=(3,0)),
            Conv(hid3,out3,k=3,s=2,p=0)
        )
        self.branch4=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
    def forward(self,x):
        return torch.cat((self.branch2(x),self.branch3(x),self.branch4(x)),1)


class multi(nn.Module):
    def __init__(self,c_in,c_out):
        super().__init__()
        self.conv1=Conv(c_in,c_out,k=(1,3),s=1,p=(0,1))
        self.conv2=Conv(c_in,c_out,k=(3,1),s=1,p=(1,0))
    def forward(self,x):
        return torch.cat((self.conv1(x),self.conv2(x)),1)

class InceptionBlock3_2(nn.Module):
    def __init__(self,c_in,out1,hid2,hid3,out3,out4):
        super().__init__()
        self.branch1=Conv(c_in,out1,k=1,s=1)
        self.branch2=nn.Sequential(
            Conv(c_in,hid2,k=1,s=1),
            multi(hid2,hid2))
        self.branch3=nn.Sequential(
            Conv(c_in,hid3,k=1,s=1),
            Conv(hid3,out3,k=3,s=1,p=1),
            multi(out3,out3)
        )
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            Conv(c_in,out4,k=1,s=1))
    def forward(self,x):
        return torch.cat((self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)),1)
    
class Inceptionv3(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes=num_classes
        self.conv1=Conv(3,32,k=3,s=2,p=0)
        self.conv2=Conv(32,32,k=3,s=1,p=0)
        self.conv3=Conv(32,64,k=3,s=1,p=1)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv4=Conv(64,80,k=3,s=1,p=0)
        self.conv5=Conv(80,192,k=3,s=2,p=0)
        self.conv6=Conv(192,288,k=3,s=1,p=1)
        # self.pool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception1_1=InceptionBlock1(288,64,48,64,64,96,32)
        self.inception1_2=InceptionBlock1(256,64,48,64,64,96,64)
        self.inception1_3=InceptionBlock1(288,64,48,64,64,96,64)

        self.inception2_1=InceptionBlock2_1(288,384,64,96,stride=2)
        self.inception2_2=InceptionBlock2_2(768,192,160,192,160,192,192)
        self.inception2_3=InceptionBlock2_2(768,192,160,192,160,192,192)
        self.inception2_4=self.inception2_3
        self.inception2_5=InceptionBlock2_2(768,192,192,192,192,192,192)
        
        self.inception3_1=InceptionBlock3_1(768,192,320,192,192)
        self.inception3_2=InceptionBlock3_2(1280,320,384,448,384,192)
        self.inception3_3=InceptionBlock3_2(2048,320,384,448,384,192)
        self.pool3=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Linear(2048,self.num_classes)
        self.softmax=nn.Softmax()

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.pool1(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.inception1_1(x)
        x=self.inception1_2(x)
        x=self.inception1_3(x)
        x=self.inception2_1(x)
        x=self.inception2_2(x)
        x=self.inception2_3(x)
        x=self.inception2_4(x)
        x=self.inception2_5(x)
        x=self.inception3_1(x)
        x=self.inception3_2(x)
        x=self.inception3_3(x)
        x=self.pool3(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        x=self.softmax(x)
        return x
    
#定义一个钩子来获取每层的输出尺寸
def print_shapes(module,input,output):
    print(f"{module.__class__.__name__.ljust(20)} |" f"Input shape : {str(input[0].shape).ljust(30)} | " f"Output shape : {str(output.shape)}")



if __name__=="__main__":
    model=Inceptionv3(1000)
    # print(model)
    #注册钩子
    hook_handles=[]
    for layer in model.children():
        handle=layer.register_forward_hook(print_shapes)
        hook_handles.append(handle)
    
    #使用模型进行前向传播
    dummy_input=torch.randn(1,3,299,299)
    with torch.no_grad():
        model(dummy_input)
    
    #移除钩子
    for handle in hook_handles:
        handle.remove()