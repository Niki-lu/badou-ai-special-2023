import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch
import models
import torch.utils.tensorboard as tensorboard
#parameters
lr=0.01
batch_size=16
num_epoches=10
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4,0.5,0.4],
        std=[0.2,0.2,0.2]
    )
])
writer=tensorboard.SummaryWriter('Resnet50.log')
model=models.Resnet50(100)
model.to(device)
criterion=torch.nn.CrossEntropyLoss()
optim=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0001)

train_data=CIFAR100(root='data',train=True,transform=transform,download=True)
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data=CIFAR100(root='./data',train=False,transform=transform,download=True)
test_loader=DataLoader(test_data,batch_size=batch_size)

#train
global_step=0
for epoch in range(num_epoches):
    running_loss=0.
    for datas,labels in train_loader:
        print("start to train")
        datas,labels=datas.to(device),labels.to(device)
        outputs=model(datas)
        loss=criterion(outputs,labels)
        loss.backward()
        optim.step()
        global_step+=1
        writer.add_scalar('loss',loss.item(),global_step)
        running_loss+=loss
    print(f'epoch {epoch} loss is {running_loss/len(train_loader)}')
model.save(model,'Resnet50.pth')

#test
with torch.no_grad():
    correction,total=0,0
    for datas,labels in test_loader:
        datas,labels=datas.to(device),labels.to(device)
        outputs=model(datas)
        correction+=(outputs==labels).sum().item()
        total+=len(datas)
    print(f"accuracy is {correction/total}")




    
