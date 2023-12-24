import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
class SimpleFCNet(nn.Module):
    def __init__(self,input_ndoes,hidden_nodes,output_nodes) -> None:
        super(SimpleFCNet,self).__init__()
        self.input_nodes=input_ndoes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        self.model=nn.Sequential(
            nn.Linear(self.input_nodes,self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes,self.output_nodes),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.model(x)
        return x
    
if __name__=="__main__":
    writer=SummaryWriter('cifar.log')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model
    input_size=32*32*3
    hidden_size=128
    output_size=10
    model=SimpleFCNet(input_size,hidden_size,output_size)
    model.to(device=device)
    #params
    lr=0.01
    
    batch_size=32
    epochs=100
    criterion=nn.CrossEntropyLoss()
    optim=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    scheduler=torch.optim.lr_scheduler.StepLR(optim,step_size=10,gamma=0.5)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45,0.46,0.43],std=[0.22,0.23,0.24])
    ])
    #data
    train_data=datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
    test_data=datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)
    train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False)
    global_step=0
    #train
    for epoch in range(epochs):
        run_loss=0.
        scheduler.step()
        for data,label in train_loader:
            data,label=data.to(device),label.to(device)
            output=model(data)
            loss=criterion(output,label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            run_loss+=loss
            global_step+=1
            writer.add_scalar('loss',loss.item(),global_step=global_step)
        #valid
        # if epoch%2==0:
            
        print(f'{epoch} loss is {run_loss/len(train_loader)}')
    print('done')
    
    #test
    correct,total=0,0
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            output=model(data)
            _,predicted=torch.max(output.data,1)
            total+=label.size(0)
            correct+=(predicted==label).sum().item()
    accuracy=correct/total
    print(f'test accuracy:{accuracy*100:.2f}%')
        
        






        
