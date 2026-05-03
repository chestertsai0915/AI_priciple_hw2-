import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import itertools

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
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def fit(self):
        print(f" 開始訓練 | Batch Size: {self.config.batch_size} | Learning Rate: {self.config.learning_rate}")
        
        # 建立字典來儲存歷史紀錄，方便稍後畫圖
        history = {
            'train_loss': [],
            'test_acc': []
        }
        
        for epoch in range(self.config.epochs):
            # --- 訓練階段 ---
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

            # --- 測試階段 ---
            self.model.eval()
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

            # 記錄到 history 中
            history['train_loss'].append(avg_train_loss)
            history['test_acc'].append(accuracy)

            # 為了版面乾淨，網格搜尋時我們可以只印出最後一個 epoch 的結果
            if epoch == self.config.epochs - 1:
                print(f"完成! 最終 Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%")
                
        return history


if __name__ == "__main__":
    # ==========================================
    # 實驗設定區 (你可以自由新增想測試的數值)
    # ==========================================
    test_batch_sizes = [256, 1024]           # 測試兩種批次大小
    test_learning_rates = [0.001, 0.0001]    # 測試兩種學習率
    fixed_epochs = 50                      # 為了節省時間，實驗時可以先跑 20 個 epoch 觀察趨勢
    
    # 準備存放所有實驗結果的字典
    all_results = {}

    # 定義資料前處理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 載入資料集 (DataLoader 會在迴圈內根據 batch_size 重新建立)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # ==========================================
    # 執行網格搜尋迴圈 (Grid Search)
    # ==========================================
    # itertools.product 會自動幫我們組合出所有可能性 (2x2=4種組合)
    for bs, lr in itertools.product(test_batch_sizes, test_learning_rates):
        experiment_name = f"BS={bs}_LR={lr}"
        
        # 1. 建立專屬此組合的 Config
        config = ModelConfig(
            epochs=fixed_epochs,
            batch_size=bs,
            learning_rate=lr,
            dropout_rate=0.2
        )
        torch.manual_seed(config.seed)

        # 2. 建立專屬此 batch_size 的 DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # 3. 初始化全新的模型與訓練器
        model = SimpleNN(config) # 如果你要測 ResNet，這裡改成 SimpleResNet(config)
        trainer = ModelTrainer(model, config, trainloader, testloader)

        # 4. 執行訓練並取得紀錄
        history = trainer.fit()
        
        # 5. 儲存結果
        all_results[experiment_name] = history

    # ==========================================
    # 繪製表現比較圖
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid') # 使用好看的圖表風格
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 畫出所有組合的 Loss 曲線
    for name, history in all_results.items():
        axes[0].plot(history['train_loss'], label=name, marker='o', markersize=4)
    axes[0].set_title('Training Loss Convergence')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # 畫出所有組合的 Accuracy 曲線
    for name, history in all_results.items():
        axes[1].plot(history['test_acc'], label=name, marker='s', markersize=4)
    axes[1].set_title('Test Accuracy Progression')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()