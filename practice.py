# practice.py
# dict_a = {"a" : 1, "b" : 2}
# print(type(dict_a.items()))
# print(dict_a.items())

# print(type(dict_a.values()))
# print(dict_a.values().__iter__())
# print(type(dict_a.keys()))
# print(dict_a.keys())

# class Country:
#     name = "미입력"
#     population = "미입력"

#     def show(self):
#         print("부모 클래스 출력입니다.")

# class Korea(Country):
#     def __init__(self, name, population):
#         self.name = name
#         self.population = population

#     def show_name(self):
#         print("여기는", self.name, "입니다.")

#     def show_population(self):
#         print("인구는", self.population, "입니다.")


# a = Korea("대한민국", "5000만명")
# a.show()
# a.show_name()
# a.show_population()

# import numpy as np
# print(np.identity(3))

# import numpy as np
# import matplotlib.pyplot as plt

# def sigmoid(x, a):
#     return 1/(1+np.exp(-a*x))

# x = np.linspace(-3, 3, 50)
# y = sigmoid(x, 1)
# plt.plot(x, y)
# plt.grid(True)
# plt.title("Sigmoid : a = 1")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# import torch

# a = torch.arange(4.)
# print(a)
# a = torch.reshape(a, (2, 2))
# print(a)

# import torch
# a = torch.tensor([[1, 2, 3], [4, 5, 6]])
# b = 3

# print(a*b)

# import torch
# import numpy as np

# x_train = torch.tensor([[1.], [2.], [3.], [4.]])
# y_train = torch.tensor([[3.], [5.], [7.], [9.]])

# W = torch.tensor(1., requires_grad = True).float()
# B = torch.tensor(1., requires_grad = True).float()
# epochs = 500
# lr = 0.01

# history = np.zeros((0, 2))

# def pred(X):
#     return W*X + B

# def mse(Yp, Y):
#     return ((Yp-Y)**2).mean()

# for i in range(epochs):
#     Yp = pred(x_train)
#     loss = mse(Yp, y_train)

#     loss.backward()

#     with torch.no_grad():
#         W -= lr *W.grad
#         B -= lr *B.grad
#         W.grad.zero_()
#         B.grad.zero_()

#     if(i % 100 == 0):
#         item = np.array([i, loss.item()])
#         history = np.vstack([history, item])

# print(f"초기상태 손실 : {history[0, 1]:.4f}")
# print(f"최종상태 손실 : {history[-1, 1]:.4f}")
# TP = 65
# FP = 7
# FN = 5
# TN = 23
# accuracy = (TP+TN)/(TP+FP+FN+TN)
# precision = (TP)/(TP+FP)
# recall = (TP)/(TP+FN)
# f1_score = 2*precision*recall/(precision + recall)
# print(f"{accuracy:.3f}")
# print(f"{precision:.3f}")
# print(f"{recall:.3f}")
# print(f"{f1_score:.3f}")

TP = 125
FP = 15
FN = 25

precision = TP/(TP + FP)
recall = TP/(TP + FN)
f1_score = 2*precision*recall/(precision + recall)
print(f"pre : {precision:.3f}, rec : {recall:.3f}, f1_s : {f1_score:.3f}")
