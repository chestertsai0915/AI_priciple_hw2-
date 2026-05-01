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

class ResidualBlock(nn.Module):
    """
    標準的殘差區塊 (Basic Block)
    包含兩層 3x3 卷積，以及一條跳躍連接 (Skip Connection)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 主路徑 (F(x)) 的第一層卷積
        # 注意：如果有使用 BatchNorm，Conv2d 的 bias 可以設為 False 以節省記憶體
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 主路徑 (F(x)) 的第二層卷積 (這層的 stride 永遠是 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 高速公路 (Skip Connection) 的維度匹配
        self.shortcut = nn.Sequential()
        # 如果輸入和輸出的維度不同，或者長寬改變了 (stride != 1)
        # 我們必須用一個 1x1 的卷積來調整高速公路的寬度，否則會無法相加
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 1. 備份原始輸入 (開上高速公路)
        identity = self.shortcut(x) 
        
        # 2. 走崎嶇的主路徑 (計算 F(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 3. 最關鍵的一步：主路徑與高速公路匯合！ F(x) + x
        out += identity 
        
        # 4. 相加完之後再做最後一次 ReLU
        out = self.relu(out) 
        return out
    

class DeepResNet(nn.Module):
    """
    大幅增加神經元數量與層數的 ResNet，專為 CIFAR-10 尺寸 (32x32) 設計
    """
    def __init__(self, config): 
        super(DeepResNet, self).__init__()
        
        # 初始特徵提取：提升到 64 個神經元 (Channels)
        self.in_channels = 64
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 動態堆疊深層網路 (4 個 Stage，每個 Stage 有多個 Block)
        # 這裡我們設定每個 Stage 有 2 個 Block，神經元數量一路飆升到 512
        self.layer1 = self._make_layer(out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, num_blocks=2, stride=2)
        
        # 全局平均池化，將任何尺寸的特徵圖壓縮成 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 業界標準的 ResNet 分類器：
        # 深層網路的特徵已經極度豐富 (512維)，不需要再經過複雜的全連接層
        # 直接一層 Linear 輸出 10 個類別，不僅參數更少，還能有效防止 Overfitting
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 10)
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        這是一個工廠方法 (Factory Method)，負責自動幫我們串接殘差區塊。
        只有第一個區塊需要降維 (stride)，後面的區塊都保持維度不變。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels # 更新輸入維度給下一個區塊
        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. 預處理
        x = self.prep(x)
        
        # 2. 穿越 18 層的殘差高速公路
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 3. 池化與分類
        x = self.pool(x)
        x = self.classifier(x)
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

    model = DeepResNet(config)
    trainer=ModelTrainer(model, config, trainloader=trainloader, testloader=testloader)
    trainer.fit()
