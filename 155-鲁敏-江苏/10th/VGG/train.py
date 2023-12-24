import torch
import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import models
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard

#parameters
batch_size=16
epochs=10
lr=0.01
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.2,0.2,0.2]
    )
])
writer=tensorboard.SummaryWriter('VGG.log')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=models.VGG(100)
model.to(device)
criterion=torch.nn.CrossEntropyLoss()
optim=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=0.0001)

train_data=CIFAR100(root='./data',train=True,transform=transform)
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)

global_step=0
for epoch in range(epochs):
    for datas,labels in train_loader:
        print("start to train")
        datas,labels=datas.to(device),labels.to(device)
        outputs=model(datas)
        loss=criterion(outputs,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        writer.add_scalar('loss',loss.item(),global_step)
torch.save(model,'VGG.pth')

