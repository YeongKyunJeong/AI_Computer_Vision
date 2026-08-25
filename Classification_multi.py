# Classification_multi.py

import numpy as np
import torch
from torch import nn, optim
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==================== Data preprocessing ==================== 

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Malgun Gothic"
np.set_printoptions(suppress = True, precision = 4)

iris = load_iris()
x_org, y_org = iris.data, iris.target

# print(x_org.shape)
# print(y_org.shape)
# print(type(x_org))
# print(type(x_org))

x_select = x_org[:, [0, 2]]

x_train, x_test, y_train, y_test = train_test_split(
    # x_select, y_org,
    x_org, y_org,    
    train_size = 75, test_size = 75, random_state = 123
    # , stratify = y_org
)

# ==================== Partition by class ==================== 
# x_t0 = x_train[y_train == 0]
# x_t1 = x_train[y_train == 1]
# x_t2 = x_train[y_train == 2]

# plt.scatter(x_t0[:, 0], x_t0[:, 1], marker = 'x', c = 'k', s = 50, label = "0 (setosa)")
# plt.scatter(x_t1[:, 0], x_t1[:, 1], marker = "o", c = 'b', s = 50, label = "1 (versicolor)")
# plt.scatter(x_t2[:, 0], x_t2[:, 1], marker = "+", c = 'k', s = 50, label = "2 (virginica)")
# plt.xlabel('sepal_length')
# plt.xlabel('petal_length')
# plt.grid(True)
# plt.legend()
# plt.show()

# ==================== Model definition ==================== 
n_input = x_train.shape[1]
n_output = len(list(set(y_train)))
# n_output = len(np.unique(y_train))
# print(np.unique(y_train, return_counts = True))
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

input_train = torch.tensor(x_train_scaled, dtype = torch.float32)
label_train = torch.tensor(y_train, dtype = torch.long)

input_test = torch.tensor(x_test_scaled, dtype = torch.float32)
label_test = torch.tensor(y_test, dtype = torch.long)

# 단순 선형 모델
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.l1 = nn.Linear(n_input, n_output)

        # to match with textbook
        self.l1.weight.data.fill_(1.0)
        # nn.init.constant_(self.l1.weight, 1.0)
        self.l1.bias.data.fill_(1.0)
        # nn.init.constant_(self.l1.bias, 1.0)

    def forward(self, x):
        return self.l1(x)

net = Net(n_input, n_output)

# for parameter in net.named_parameters():
#     print(parameter)

lr = 1e-2
crit = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = lr)
num_epochs = 10000
# sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 2e-2, steps_per_epoch = 1, epoch = num_epochs)

history = np.zeros((0, 5))

for epoch in range(num_epochs):
    net.train()
    optimizer.zero_grad()
    output = net(input_train)
    loss = crit(output, label_train)

    loss.backward()
    optimizer.step()
    # sched.step()

    predicted = torch.max(output, 1)[1]

    train_loss = loss.item()
    # train_acc = (predicted == label_train).sum() / len(label_train)
    train_acc = (predicted == label_train).float().mean().item()

    with torch.no_grad():
        net.eval()
        output_test = net(input_test)
        val_loss = crit(output_test, label_test)
        predicted_test = torch.max(output_test, 1)[1]
        # val_acc = (predicted_test == label_test).sum() / len(label_test)
        val_acc = (predicted_test == label_test).float().mean().item()

    if(epoch % 1000 == 0):
        print(f"Epoch {epoch}/{num_epochs}, loss : {train_loss:.5f}, acc : {train_acc:.5f}, val_loss : {val_loss:.5f}, val_acc : {val_acc:.5f}")
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

plt.figure(figsize = (13, 6))
fig = plt.subplot(1, 2, 1)
plt.plot(history[:,0], history[:,1], 'b', label='훈련')
plt.plot(history[:,0], history[:,3], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.legend()

fig = plt.subplot(1, 2, 2)
plt.plot(history[:,0], history[:,2], 'b', label='훈련')
plt.plot(history[:,0], history[:,4], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()