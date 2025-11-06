"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪个维度的值最大，就属于哪个类别
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 创建模型
class Torchmodel(nn.Module):
    def __init__(self,input_size,output_size):
        super(Torchmodel, self).__init__()
        self.linear=nn.Linear(input_size,output_size)
        self.loss=nn.CrossEntropyLoss()     # 交叉熵自带softmax

    def forward(self,x,y=None):
        y_pre=self.linear(x)
        if y:
            loss=self.loss(y_pre,y)
            return y_pre,loss
        else:
            return y_pre


# 随机生成训练样本，构造数据
class TrainDataset(Dataset):
    def __init__(self,train_num_samples,nun_vector):
        self.nun_vector = nun_vector
        self.num_samples=train_num_samples
        self.X,self.Y=self._generate_sample()

    def _generate_sample(self):
        X=[]
        Y=[]
        for _ in range(self.num_samples):
            x=np.random.randn(self.nun_vector)
            y=np.argmax(x)
            X.append(x)
            Y.append(y)
        return torch.FloatTensor(X),torch.LongTensor(Y)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]


# 随机生成测试样本，构造数据
class TsetDataset(Dataset):
    def __init__(self, test_num_samples,nun_vector):
        self.nun_vector=nun_vector
        self.num_samples = test_num_samples
        self.X, self.Y = self._generate_sample()

    def _generate_sample(self):
        X = []
        for _ in range(self.num_samples):
            x = np.random.randn(self.nun_vector)
            X.append(x)
        return torch.FloatTensor(X)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx]



def train_model_process(model,train_dataloader,num_epochs):
    # 设定训练设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 指定优化器,损失函数
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.CrossEntropyLoss()
    # 将模型放入设备中
    model=model.to(device)

    for epoch in range(num_epochs):
        train_num=0
        train_corrects = 0
        watch_loss = []
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
            # 将训练数据放入设备中
            b_x=batch_x.to(device)
            b_y=batch_y.to(device)

            model.train()
            # 前向传播,计算损失
            y_pre,loss=model(b_x,b_y)
            pre_label=torch.argmax(torch.softmax(y_pre,dim=1),dim=1)

            # 反向传播，梯度更新，清空梯度
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 记录损失
            watch_loss.append(loss.item())
            train_corrects+=torch.sum(pre_label==b_y)
            train_num += b_x.size(0)
        print(f"epoch:{epoch},loss:{np.mean(watch_loss)}")
        print(f"train_num:{train_num},train_corrects:{train_corrects}")


model=Torchmodel(5,5)

train_data=TrainDataset(train_num_samples=1000,nun_vector=5)

dataloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=2  # 多进程加载数据
)



train_model_process(model,dataloader,20)





