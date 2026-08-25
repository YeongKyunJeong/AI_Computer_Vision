# Gradient descent.py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

np.set_printoptions(suppress = True)
# suppress=True # : 가능하면 과학적 표기법(1e3)을 쓰지 않음

def L(u, v):
    return 3*u **2 + 3*v**2 - u*v + 7*u - 7*v + 10
def Lu(u, v):
    return 6*u - v + 7
def Lv(u, v):
    return 6*v - u - 7

u = np.linspace(-5, 5, 501)
v = np.linspace(-5, 5, 501)
U, V = np.meshgrid(u, v)
Z = L(U, V)

W = np.array([4., 4.])
W1 = [W[0]]
W2 = [W[1]]
N = 21
alpha = 0.05
for i in range(N):
    W = W - alpha * np.array([Lu(W[0], W[1]), Lv(W[0], W[1])])
    W1.append(W[0])
    W2.append(W[1])
n_loop = 11

WW1 = np.array(W1[:n_loop])
WW2 = np.array(W2[:n_loop])
ZZ = L(WW1, WW2)

fig = plt.figure(figsize = (10 ,10))
ax = plt.axes(projection = "3d") # 3차원 좌표계
ax.set_zlim(0, 250)
ax.set_xlabel("W")
ax.set_xlabel("B")
ax.set_zlabel("loss")
ax.view_init(50, 240)  # 어느 방향에서 바라볼지, elev, azim 순
# ax.xaxis._axinfo["grid"]['linewidth'] = 2. # grid 선 굵기
# ax.yaxis._axinfo["grid"]['linewidth'] = 2.
# ax.zaxis._axinfo["grid"]['linewidth'] = 2.
ax.contour3D(U, V, Z, 100, cmap = "Blues", alpha = 0.7)
ax.plot3D(WW1, WW2, ZZ, 'o--', c='k', alpha = 1, markersize = 5)
plt.show()
