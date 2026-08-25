# Scheduler_and_hook.py

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# device = 'cuda' if torch.cuda.is_avaliable() else 'cpu'

tfm = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root = r'C:\ROKEY\deep_learning\MNIST', train = True, download = True, transform = tfm)
test_ds = datasets.MNIST(root = r'C:\ROKEY\deep_learning\MNIST', train = False, download = True, transform = tfm)

# print(type(train_ds))
# print(len(train_ds))
# print(train_ds[0][0].shape) # 이미지 (1, 28, 28)
# print(train_ds[0][1])       # 라벨

# image, label = train_ds[0]

train_loader = DataLoader(train_ds, batch_size = 256, shuffle = True, num_workers = 0, pin_memory = True)
test_loader = DataLoader(test_ds, batch_size = 512, shuffle = False, num_workers = 0, pin_memory = True)

x, y = next(iter(train_loader))

print("한 batch의 x:", x.shape)
print("한 batch의 y:", y.shape)
print("batch 첫 이미지:", x[0].shape)
print("batch 첫 라벨:", y[0])

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10) # 출력 : 10개의 클래스
        )

        for m in self.f:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.f(x)

# model = MLP().to(device)
model = MLP()

# ================= Lr scheduler ====================
opt = optim.AdamW(model.parameters(), lr = 3e-3)
sched = optim.lr_scheduler.OneCycleLR(opt, max_lr = 3e-3, steps_per_epoch = len(train_loader), epochs = 5)
crit = nn.CrossEntropyLoss()   # soft max가 포함되어 모델에 soft max 변환을 넣으면 안 됨

# ================= hook ====================
grads = []

# def hook_fn(m, gi, go):
#     if isinstance(m, nn.Linear):
#         if m.weight.grad is not None:
#             grads.append(m.weight.grad.detach().abs().mean().item())

# hooks = [m.register_backward_hook(hook_fn) for m in model.modules() if isinstance(m, nn.Linear)]

def hook_fn(grad):
    grads.append(grad.detach().abs().mean().item())

hooks = [ m.weight.register_hook(hook_fn) for m in model.modules() if isinstance(m, nn.Linear)]

def train_epoch():
    model.train()
    tot = 0
    correct = 0

    for x, y in train_loader:
        # x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)

        loss.backward()
        opt.step()
        sched.step()

        tot += y.size(0)
        # tot += y.size()[0]
        # tot += y.shape[0]
        correct += (out.argmax(1) == y).sum().item() # argmax는 역전파 관련 연산이 없기 때문에 detach 필요X
    
    return loss.item(), correct/tot

def eval_epoch():
    model.eval()

    tot = 0
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            # x, y = x.to(device), y.to(device)
            out = model(x)
            # loss를 비교하는게 아니라 얼마나 맞았는지를 평가해야 함
            tot += y.size(0)
            correct += (out.argmax(1) == y).sum().item()

    return correct/tot

hist_grad = []
hist_acc = []

for epoch in range(5):
    loss, tr_acc = train_epoch()
    acc = eval_epoch()

    hist_grad.append(sum(grads[-len(train_loader)*3:]) / max(1, 3*len(train_loader)))

    hist_acc.append(acc)

    print(f"epoch = {epoch + 1} loss = {loss:.3f} test_acc = {acc:.3f}")

plt.figure(figsize = (13, 6))
fig = plt.subplot(1, 2, 1)
plt.plot(hist_grad)
plt.title('average |grad|')

fig = plt.subplot(1, 2, 2)
plt.plot(hist_acc)
plt.title('test_acc')
plt.show()

for h in hooks:
    h.remove()