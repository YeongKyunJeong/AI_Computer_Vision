# Deep_learning_practice.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
# from torchviz import make_dot
# from IPython.display import display

plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["axes.grid"] = True
np.set_printoptions(suppress = True)

# =========================== Data preprocessing ===========================

sample_data1 = np.array([
    [166.0, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
    ])

# print(sample_data1)

x = sample_data1[:, 0]
y = sample_data1[:, 1]

# plt.scatter(x, y, c = "k", marker = "o", s = 50)
# plt.xlabel("X: 신장(cm)")
# plt.ylabel("Y: 체중(cm)")
# plt.title("신장과 체중의 관계 분포")
# plt.show()

X = x - x.mean()
Y = y - y.mean()

X_tensor = torch.tensor(X, dtype = torch.float32)
Y_tensor = torch.tensor(Y, dtype = torch.float32)

W = torch.tensor(1.0, requires_grad = True, dtype = torch.float32)
B = torch.tensor(1.0, requires_grad = True, dtype = torch.float32)

# =========================== Prediction function ===========================

def pred(X):
    return W*X + B

Yp = pred(X_tensor)
# print(Yp)

# params = {"W" : W, "B" : B}
# g = make_dot(Yp, params = params)
# # display(g)

# =========================== Loss function ===========================

def mse(Yp, Y):
    return ((Yp - Y)**2).mean()

# loss = mse(Yp, Y_tensor)
# loss.backward()

# # print(W.grad)
# # print(B.grad)

# =========================== Update parameters ===========================

# lr = 0.001

# # requires_grad = True인 텐서를 직접 값을 바꾸려고 하면 에러
# # try:
# #     W -= lr*W.grad
# #     B -= lr*B.grad
# # except RuntimeError as e:
# #     # raise e
# #     print(e)

# with torch.no_grad():
#     W -= lr*W.grad
#     B -= lr*B.grad

# W.grad.zero_()
# B.grad.zero_()
# print(W)
# print(B)

# Repeatation

# history = np.zeros((0, 2))
# num_epochs = 500
# lr = 0.001

# for epoch in range(num_epochs):

#     Yp = pred(X_tensor)
#     loss = mse(Yp, Y_tensor)
#     loss.backward()
#     with torch.no_grad():
#         W -= lr*W.grad
#         B -= lr*B.grad

#     W.grad.zero_()
#     B.grad.zero_()

#     if(epoch % 10 == 0):
#         item = np.array([epoch, loss.item()])
#         history = np.vstack((history, item)) # 1차원 벡터가 들어오면 알아서 행으로 처리함
#         # item = np.array([[epoch, loss.item()]])
#         # history = np.concat((history, item), axis = 0)
#         # print(f"{epoch:3.0f} : loss = {loss:6.3f}")

# # Analysis results
# plt.figure(figsize = (12, 6))
# fig = plt.subplot(1, 2, 1)
# plt.plot(history[:,0], history[:,1], c = 'b')
# plt.xlabel("반복 횟수")
# plt.ylabel("loss")
# plt.title("경사하강법 학습 곡선")

# fig = plt.subplot(1, 2, 2)
# plt.scatter(X, Y, c = 'k', s = 50)
# X_range = torch.from_numpy(np.array([X.min(), X.max()])).float()
# Y_range = pred(X_range)

# plt.scatter(X, Y, c='k', s=50) # 원본 데이터의 산포도 (검은 점)
# plt.plot(X_range.data,Y_range.data, c = 'c')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("신장과 체중의 관계")
# plt.tight_layout()
# plt.show()

# =========================== Update parameters with optimizer ===========================

history_default = np.zeros((0, 2))
history_momentum = np.zeros((0, 2))
num_epochs = 500
lr = 0.001


# for epoch in range(num_epochs):

#     Yp = pred(X_tensor)
#     loss 




for epoch in range(num_epochs):
    Yp = pred(X_tensor)

    loss = mse(Yp, Y_tensor)
    loss.backward()
    with torch.no_grad():
        W -= lr*W.grad
        B -= lr*B.grad

    W.grad.zero_()
    B.grad.zero_()

    if(epoch % 10 == 0):
        item1 = np.array([[epoch, loss.item()]])
        history_default = np.vstack((history_default, item1))

W = torch.tensor(1.0, requires_grad = True, dtype = torch.float32)
B = torch.tensor(1.0, requires_grad = True, dtype = torch.float32)

optimizer = optim.SGD([W, B], lr = lr, momentum = 0.9)
for epoch in range(num_epochs):
    Yp = pred(X_tensor)

    loss = mse(Yp, Y_tensor)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    if(epoch % 10 == 0):
        
        item2 = np.array([[epoch, loss.item()]])
        history_momentum = np.concat((history_momentum, item2), axis = 0)


plt.plot(history_default[:,0], history_default[:,1], 'c', label='기본값 설정')
plt.plot(history_momentum[:,0], history_momentum[:,1], 'k', label='momentum=0.9')
plt.xlabel('반복 횟수')
plt.ylabel('loss')
plt.legend()
plt.title('학습 곡선(손실)')
plt.show()