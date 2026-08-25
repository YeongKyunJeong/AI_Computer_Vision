# Model_complexity.py

import numpy as np
import torch, math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

N = 600
X = torch.linspace(-3*math.pi, 3*math.pi, 600, dtype = torch.float).unsqueeze(1)
# X = torch.linspace(-3*math.pi, 3*math.pi, 600, dtype = torch.float).view(-1, 1)
y = torch.sin(X) + 0.2*torch.randn_like(X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

# X_train = torch.tensor(X_train, dtype = torch.float32)
# y_train = torch.tensor(y_train, dtype = torch.float32)
# X_test = torch.tensor(X_test, dtype = torch.float32)
# y_test = torch.tensor(y_test, dtype = torch.float32)
# X_val = torch.tensor(X_val, dtype = torch.float32)
# y_val = torch.tensor(y_val, dtype = torch.float32)

# X_train = X_train.detach().clone()
# y_train = y_train.detach().clone()
# X_test = X_test.detach().clone()
# y_test = y_test.detach().clone()
# X_val = X_val.detach().clone()
# y_val = y_val.detach().clone()

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(X_val.shape)
# print(y_val.shape)

def make_mlp(hidden):
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1)
    )

small = make_mlp(hidden = 8)
big = make_mlp(hidden = 128)

# def train(model, Xtr, ytr, Xva, yva, epochs = 600, lr = 1e-3):
#     opt = optim.Adam(model.parameters(), lr = lr)
#     loss_fn = nn.MSELoss()
#     t_hist, v_hist = np.zeros((0, 2)), np.zeros((0, 2))
#     for epoch in range(epochs):
#         model.train()
#         opt.zero_grad()
#         pred = model(Xtr)
#         loss = loss_fn(pred, ytr)
#         loss.backward()
#         opt.step()

#         model.eval()
#         with torch.no_grad():
#             v_loss = loss_fn(pred(Xva), yva)
#         t_hist = np.vstack(t_hist, np.array([epoch, loss.item()]))
#         v_hist = np.vstack(v_hist, np.array([epoch, v_loss.item()]))

def train(model, Xtr, ytr, xval, yval, epochs = 600, lr = 1e-3):
    t_hist, v_hist = np.zeros((0,2)), np.zeros((0, 2))
    opt = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = criterion(pred, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(xval), yval)
        t_hist = np.vstack((t_hist, np.array([epoch, loss.item()])))
        v_hist = np.vstack((v_hist, np.array([epoch, v_loss.item()])))

    return t_hist, v_hist

tr_s, va_s = train(small, X_train, y_train, X_val, y_val, epochs = 300, lr = 1e-3)
tr_b, va_b = train(big, X_train, y_train, X_val, y_val, epochs = 300, lr = 1e-3)

plt.figure();
plt.plot(tr_s[:, 0], tr_s[:, 1], label='small-train')
plt.plot(va_s[:, 0], va_s[:, 1], label='small-val')
plt.plot(tr_b[:, 0], tr_b[:, 1], label='big-train')
plt.plot(va_b[:, 0], va_b[:, 1], label='big-val')
plt.legend(); plt.title('Bias-Variance')
plt.show()