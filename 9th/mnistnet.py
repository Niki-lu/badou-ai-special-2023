import torch 
import torch.nn as nn
import torch.nn.functional as functional
import torchvision 
import torchvision.transforms as transforms


class Model:
    def __init__(self,net,cost,optimizer) -> None:
        self.net=net
        self.cost=self.create_cost(cost)
        self.optimizer=self.create_optimizer(optimizer)
    
    def create_cost(self,cost):
        support_cost={
            'CROSS_ENTROPY':nn.CrossEntropyLoss(),
            'MSE':nn.MSELoss()
        }
        return support_cost[cost]
    
    def create_optimizer(self,optimizer,**rest):
        support_optimizer={
            "SGD":torch.optim.SGD(self.net.parameters(),lr=0.1,**rest),
            "RMSprop":torch.optim.RMSprop(self.net.parameters(),lr=0.01,**rest),
            "Adam":torch.optim.Adam(self.net.parameters(),lr=0.01,**rest)
        }
        return support_optimizer[optimizer]
    
    def train(self,train_loader,epochs=3):
        for epoch in range(epochs):
            running_loss=0.0
            for i,data in enumerate(train_loader,0):
                self.optimizer.zero_grad()
                inputs,labels=data
                outputs=self.net(inputs)
                loss=self.cost(labels,outputs)
                loss.backward()
                self.optimizer.step()

                running_loss+=loss
                if i%100==0:
                    print('epoch %d %.2f loss: %.3f'%(epoch+1,(i+1)*1./len(train_loader),running_loss/100))
                    running_loss=0.0
    
    def evaluate(self,test_loader):
        print("start evaluate...")
        correct,total=0,0
        with torch.no_grad():
            for data in test_loader:
                imgs,labels=data
                output=self.net(imgs)
                numbers=torch.argmax(output,1)
                total+=labels.size(0)
                correct+=(numbers==labels).sum().item()
        print('Accuracy is : %d %%'%(100*correct/total))

def mnist_load_data():
    transforms=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,],[1,])]
    )
    
    trainset=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transforms)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=4)
    testset=torchvision.datasets.MNIST(root='./data',train=False,transform=transforms,download=True)
    testloader=torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,num_works=4)
    return trainloader,testloader

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1=nn.Linear(28*28,512)
        self.fc2=nn.Linear(512,512)
        self.fc3=nn.Linear(512,10)
        
    def forward(self,x):
        x=x.view(-1,28*28)
        x=function.Relu(self.fc1(x))
        x=function.Relu(self.fc2(x))
        x=function.softmax(self.fc3(x),dim=1)
        return x
    
if __name__=='__main__':
    net=MnistNet()
    model=Model(net,'CROSS_ENTROPY','RMSP')
    trainloader,testloader=mnist_load_data()
    model.train(trainloader)
    model.evaluate(testloader)

        
