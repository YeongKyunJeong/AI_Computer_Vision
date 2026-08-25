# MLP_practice.py

# import torch
# import torch.optim as optim
# import numpy as np
# import matplotlib as plt

# X = torch.linspace(10, 20, 101).float()
# Y = 0.5 * X + 0.2* torch.randn_like(X)
# # Xn = X - X.mean()
# # Yn = Y - Y.mean()
# std_X = X.std()
# mean_X = X.mean()
# std_Y = Y.std()
# mean_Y = Y.mean()
# Xn = (X - mean_X)/std_X
# Yn = (Y - mean_Y)/std_Y
# print(f"x_mean : {mean_X}, y_mean : {mean_Y}, x_std : {std_X}. y_std : {std_Y}")
# W = torch.tensor([1], requires_grad = True ,dtype = torch.float32)
# B = torch.tensor([2], requires_grad = True, dtype = torch.float32)
# lr = 1e-3
# optimizer = optim.SGD([W, B], lr, momentum = 0.9) 
# num_epoch = 1000
# history = np.zeros((0, 2))
# def predY(X): return W*X + B
# def mse(Yp, Y): return ((Yp - Y)**2).mean()
# for epoch in range(num_epoch):
#     Yp = predY(Xn)
#     loss = mse(Yp, Yn)
#     loss.backward()

#     optimizer.step()
#     optimizer.zero_grad()
#     # with torch.no_grad():
#     #     W -= lr*W.grad
#     #     B -= lr*B.grad

#     # W.grad.zero_()
#     # B.grad.zero_()

#     if (epoch % 10 == 0):
#         item = np.array([epoch, loss.item()])
#         history = np.vstack((history, item))
#         print(f"epoch = {epoch} loss = {loss.item():.4f}")
# print(W.item(), B.item())
# W_pred = (std_Y * W.item())/std_X
# B_pred =  std_Y * B.item() + mean_Y - W_pred*mean_X
# print(f"W_pred = {W_pred},B_pred = {B_pred} ")

# ======================Predict Function with nn.Module=======================
# import torch
# import torch.nn as nn

# class ClassifierModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ClassifierModel, self).__init__()

#         self.layer1 = nn.Linear(input_dim, 64)
#         self.relu1 = nn.ReLU()
#         self.layer2 = nn.Linear(64, 32)
#         self.relu2 = nn.ReLU()
#         self.layer3 = nn.Linear(32, output_dim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x1 = self.layer1(x)
#         x2 = self.relu1(x1)
#         x3 = self.layer2(x2)
#         x4 = self.relu2(x3)
#         x5 = self.layer3(x4)
#         out = self.sigmoid(x5)
#         return out

# model_cls = ClassifierModel(input_dim = 20, output_dim = 1)
# print(model_cls)

# test_input_cls = torch.randn(10, 20)
# output_cls = model_cls(test_input_cls)

# print(f"입력 데이터 크기 : {test_input_cls.shape}")
# print(f"출력 데이터 크기 : {output_cls.shape}")
# print(f"Sigmoid 적용 출력 : {output_cls}")

# ======================Predict Function with nn.Module & Sequential=======================
# import torch
# import torch.nn as nn

# class ClassifierModelSeq(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ClassifierModelSeq, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, output_dim),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.network(x)

# module_seq = ClassifierModelSeq(input_dim = 20, output_dim = 1)
# test_input_seq = torch.randn(10, 20)
# output_seq = module_seq(test_input_seq)

# print(f"입력 데이터 크기 : {test_input_seq.shape}")
# print(f"출력 데이터 크기 : {output_seq.shape}")
# print(f"Sigmoid 적용 출력 : {output_seq}")

# ======================Model with Sequential=======================