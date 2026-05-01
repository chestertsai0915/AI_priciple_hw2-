import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """
    管理超參數
    """
    dropout_rate: float = 0.2  #dropout率
    epochs:int =50
    batch_size:int =1024
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    device:str ="cuda" if torch.cuda.is_available() else "cpu"  #設備選擇

    seed: int = 42  #隨機種子


class SimpleCNN(nn.Module):
    def __init__(self,config:ModelConfig):
        super(SimpleCNN, self).__init__()
        
        self.features=nn.Sequential(
            #layer1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), #in_channels=3 分三層RGB in_channels=3 Padding (填充) strid步長
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #池化層

            #layer2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #池化層
            
            # layer3: 8x8 -> 4x4 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classfier=nn.Sequential(
            
            nn.Flatten(), # 將 128 個 4x4 的特徵圖攤平

            # 第一層全連接層 (128 * 4 * 4 = 2048)
            nn.Linear(128*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),

            #輸出層
            nn.Linear(512,10)   
        )
    def forward(self,x):
        x=self.features(x)
        x=self.classfier(x)
        return x
    
    

class ModelTrainer:
    def __init__(self, model: nn.Module, config: ModelConfig, trainloader, testloader):
        self.config=config
        self.device=config.device
        self.model=model.to(self.device)
        self.trainloader=trainloader
        self.testloader=testloader

        self.criterion=nn.CrossEntropyLoss()
        self.optimizer=optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def fit(self):
        print(f"開始訓練，使用設備: {self.device}")
        for epoch in range(self.config.epochs):
            self.model.train()
            runnning_loss=0.0
            for inputs, labels in self.trainloader:
                inputs, labels =inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs=self.model(inputs)
                loss=self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                runnning_loss += loss.item()
            avg_train_loss= runnning_loss/len(self.trainloader)
            
            self.model.eval()
            correct=0
            total=0

            with torch.no_grad():
                for inputs, labels in self.testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs=self.model(inputs)
                    _, predicted= torch.max(outputs.data,1)
                    total+= labels.size(0)
                    correct+=(predicted==labels).sum().item()
            accuracy=100*correct/total
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1:02d}/{self.config.epochs}] - Train Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
                
        print(" 訓練完成！")


if __name__ == "__main__":
    # 1. 設定超參數
    config = ModelConfig(
        dropout_rate = 0.2,  #dropout率
        epochs=50,
        batch_size=1024,
        learning_rate = 0.001,
        weight_decay= 0.0001,
        device="cuda" if torch.cuda.is_available() else "cpu"  #設備選擇

    )

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), # 隨機水平翻轉 (資料擴增)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f" 訓練集樣本數: {len(trainset)}, 測試集樣本數: {len(testset)}")

    model = SimpleCNN(config)
    traier=ModelTrainer(model, config, trainloader=trainloader, testloader=testloader)
    traier.fit()
