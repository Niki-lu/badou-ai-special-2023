import torch
import torch.nn as nn

class Conv(nn.Module):
    default_act=nn.ReLU()
    def __init__(self,c_in,c_out,k=1,s=1,p=0,g=1,act=True):
        super().__init__()
        self.conv=nn.Conv2d(c_in,c_out,k,s,padding=p,groups=g)
        self.bn=nn.BatchNorm2d(c_out)
        self.act=self.default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self,c_in,c_out,k=1,s=1,p=0,act=True):
        super(DWConv,self).__init__()
        self.conv1=Conv(c_in,c_in,k=k,s=s,g=c_in,p=p,act=act)
        self.conv2=Conv(c_in,c_out)
    
    def forward(self,x):
        return self.conv2(self.conv1(x))


class MobileNet(nn.Module):
    def __init__(self,num_classes) -> None:
        super().__init__()
        self.num_classes=num_classes
        self.features=nn.Sequential(
            Conv(3,32,3,2,1),
            DWConv(32,64,3,1,1),
            DWConv(64,128,3,2,1),
            DWConv(128,128,3,1,1),
            DWConv(128,256,3,2,1),
            DWConv(256,256,3,1,1),
            DWConv(256,512,3,2,1),
            *[DWConv(512,512,3,1,1) for _ in range(5)],
            DWConv(512,1024,3,2,1),
            DWConv(1024,1024,3,1,1),
        )
        self.avgpool=nn.AvgPool2d((7,7),1)
        self.fc=nn.Linear(1024,self.num_classes)
        self.softmax=nn.Softmax()
    
    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        x=self.softmax(x)
        return x

def print_shapes(module,input,output):
    print(f"{module.__class__.__name__.ljust(20)} | " f"input shape : {str(input[0].shape).ljust(30)}| " f"output shape : {str(output.shape)}")

def register_hooks(module):
    hook_handlers=[]
    for layer in model.features.children():
        handler=layer.register_forward_hook(print_shapes)
        hook_handlers.append(handler)
    return hook_handlers

if __name__=="__main__":
    model=MobileNet(1000)
    
    hook_handlers=register_hooks(model)
    
    dummy_input=torch.randn(1,3,224,224)
    with torch.no_grad():
        model(dummy_input)
    
    for handler in hook_handlers:
        handler.remove()


