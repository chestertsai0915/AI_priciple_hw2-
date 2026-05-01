import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """只管理訓練超參數，不干涉模型架構"""
    # 訓練與正則化參數
    dropout_rate: float = 0.2              # Dropout率
    epochs: int = 50                       # 訓練總輪數
    batch_size: int = 1024                 # 批次大小 (依照作業要求)
    learning_rate: float = 0.001           # 學習率
    weight_decay: float = 0.0001           # L2 正則化
    optimizer_type: str = "Adam"           # 優化器

    # 環境參數
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42                         # 隨機種子


class SimpleNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        
        # 網路架構固定寫死，清楚明瞭
        self.network = nn.Sequential(    
            # Layer 1
            nn.Linear(32 * 32 * 3, 1024),
            nn.BatchNorm1d(1024),             # 加入 BatchNorm 解決 ReLU 死亡與加速收斂
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),  # 只從 config 讀取 dropout 機率

            # Layer 2
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),

            # Layer 3
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            # Output Layer (10個類別)
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


class ModelTrainer:
    def __init__(self, model: nn.Module, config: ModelConfig, trainloader, testloader):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 根據 config 設定優化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def fit(self):
        print(f" 開始訓練，使用設備: {self.config.device}")
        
        for epoch in range(self.config.epochs):
            #訓練
            self.model.train()
            running_loss = 0.0
            
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() 
                
            avg_train_loss = running_loss / len(self.trainloader)

            # 測試
            self.model.eval() # 關閉 dropout 與 batchnorm 的訓練模式
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            # 輸出進度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1:02d}/{self.config.epochs}] - Train Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
                
        print(" 訓練完成")


if __name__ == "__main__":
    config = ModelConfig(
        epochs=50,
        batch_size=1024,
        learning_rate=0.001,
        dropout_rate=0.2
    )

    
    torch.manual_seed(config.seed)

    # 2. 定義資料前處理與載入 DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"訓練集樣本數: {len(trainset)}, 測試集樣本數: {len(testset)}")

    # 3. 初始化模型與訓練器
    model = SimpleNN(config)
    trainer = ModelTrainer(model, config, trainloader, testloader)

    # 4. 執行訓練
    trainer.fit()