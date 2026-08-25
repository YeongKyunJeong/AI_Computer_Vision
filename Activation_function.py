# Activation_function.py

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# np.random.seed(123)

x = np.random.randn(100, 1) * 2.5 # 표준편차가 2.5은 정규분포 난수 100개

y = x**2 + np.random.randn(100, 1) * 0.8

x_train = torch.tensor(x[:50, :], dtype = torch.float32)
y_train = torch.tensor(y[:50, :], dtype = torch.float32)
x_test = torch.tensor(x[50:, :], dtype = torch.float32)
y_test = torch.tensor(y[50:, :], dtype = torch.float32) 

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

# plt.scatter(x_train, y_train, c = "c", label = "훈련 데이터")
# plt.scatter(x_test, y_test, c = "k", marker = 'x', label = "검증 데이터")
# plt.legend()
# plt.show()

# class Net1(nn.Module):
#     def __init__(self):
#         super(Net1, self).__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, x):
#         return self.linear(x)

# lr = 0.01
# net = Net1()
# optimizer = optim.SGD(net.parameters(), lr = lr)
# criterion = nn.MSELoss()

# num_epoch = 1000
# history = np.zeros((0, 2))

# for epoch in range(num_epoch):

#     # net.train()   # nn.Linear는 train() 모드와 eval() 모드에서 동작이 똑같아서 생략해도 됨
#                   # 모델은 생성 직후 기본적으로 training mode
#     optimizer.zero_grad()
#     output = net(x_train)
#     loss = criterion(output, y_train)
#     loss.backward()
#     optimizer.step()

#     if(epoch % 100 == 0):
#         history = np.vstack((history, np.array([epoch, loss.item()])))
#         print(f"epoch : {epoch} loss = {loss:.5f}")

# with torch.no_grad():
    # y_pred = net(x_test)
    # print(x_test.data.shape)

# plt.title("은닉층 없음, 활성화 함수 없음")
# plt.scatter(x_test.data[:, 0], y_test.data[:, 0], c = 'k', label = "정답")
# plt.scatter(x_test.data[:, 0], y_pred.data[:, 0], c = 'b', label = "예측", marker = 'x')
# plt.legend()
# plt.show()

# class Net2(nn.Module):
#     def __init__(self):
#         super(Net2, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(1, 16),
#             nn.Linear(16, 8),
#             nn.Linear(8, 1),
#         )

#     def forward(self, x):
#         return self.network(x)

# lr = 0.01
# net2 = Net2()
# optimizer = optim.SGD(net2.parameters(), lr = lr)
# criterion = nn.MSELoss()

# num_epoch = 1000
# history = np.zeros((0, 2))

# for epoch in range(num_epoch):

#     # net2.train()

#     optimizer.zero_grad()
#     output = net2(x_train)
#     loss = criterion(output, y_train)
#     loss.backward()
#     optimizer.step()

#     if(epoch % 100 == 0):
#         history = np.vstack((history, np.array([epoch, loss.item()])))
#         print(f"epoch : {epoch} loss = {loss:.5f}")

# with torch.no_grad():
#     y_pred = net2(x_test)

# plt.title("은닉층 2개, 활성화 함수 없음")
# plt.scatter(x_test.data[:, 0], y_test.data[:, 0], c = 'k', label = "정답")
# plt.scatter(x_test.data[:, 0], y_pred.data[:, 0], c = 'b', label = "예측", marker = 'x')
# plt.legend()
# plt.show()

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.network(x)

lr = 0.01
net3 = Net3()
optimizer = optim.SGD(net3.parameters(), lr = lr)
criterion = nn.MSELoss()

num_epoch = 1000
history = np.zeros((0, 2))

for epoch in range(num_epoch):

    # net2.train()

    optimizer.zero_grad()
    output = net3(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if(epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f"epoch : {epoch} loss = {loss:.5f}")

with torch.no_grad():
    y_pred = net3(x_test)

plt.title("은닉층 2개, 활성화 함수 없음")
plt.scatter(x_test.data[:, 0], y_test.data[:, 0], c = 'k', label = "정답")
plt.scatter(x_test.data[:, 0], y_pred.data[:, 0], c = 'b', label = "예측", marker = 'x')
plt.legend()
plt.show()