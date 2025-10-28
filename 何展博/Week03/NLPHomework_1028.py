# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json

"""
基于pytorch的网络编写
实现一个多分类任务：返回文本中'我'的索引位置
文本长度固定为10，如果'我'存在多个返回所有位置，不存在返回10
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.classify = nn.Linear(vector_dim, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        batch_size, sen_len, vector_dim = x.shape
        x = x.view(batch_size * sen_len, vector_dim)

        x = self.classify(x)

        if y is not None:

            y = y.view(-1)  # (batch_size * sen_len)
            return self.loss(x, y.long())  # 计算交叉熵损失
        else:
            prob = F.softmax(x, dim=1)  # (batch_size * sen_len, 2)
            return prob[:, 1].view(batch_size, sen_len)  # (batch_size, sen_len)

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab

def build_sample(vocab, sentence_length=10):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 构建标签：10维向量，如果该位置是'我'则为1，否则为0
    y = [1 if char == '我' else 0 for char in x]
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号
    return x, y


def build_dataset(sample_length, vocab, sentence_length=10):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # y是10维向量
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length=10):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for i in range(len(y_pred)):
            pred_positions = [idx for idx, prob in enumerate(y_pred[i]) if prob >= 0.5]
            # 获取真实位置
            real_positions = [idx for idx, val in enumerate(y[i]) if val == 1]

            # 比较预测位置和真实位置
            if pred_positions == real_positions:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 10
    learning_rate = 0.005

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    # 修复警告：显式设置 weights_only=True
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        # 确保输入字符串长度为10
        if len(input_string) > sentence_length:
            input_string = input_string[:sentence_length]
        else:
            # 补充到长度10
            input_string = input_string + ' ' * (sentence_length - len(input_string))

        # 将输入序列化
        x.append([vocab.get(char, vocab['unk']) for char in input_string])

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测 (batch, 10)

    for i, input_string in enumerate(input_strings):
        # 获取预测位置
        positions = [idx for idx, prob in enumerate(result[i]) if prob >= 0.5]
        # 如果没有找到'我'，返回10
        if not positions:
            print("输入：%s, 预测位置：10" % input_string)
        else:
            print("输入：%s, 预测位置：%s" % (input_string, str(positions)))


if __name__ == "__main__":
   # main()
    # 测试用例
    test_strings = ["我爱中国", "hello我", "world", "我我我", "abcde"]
    predict("model.pth", "vocab.json", test_strings)