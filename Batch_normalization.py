# Batch_normalization.py

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# # =================== Effect by batch size =====================
# device = 'cuda' if torch.cuda.is_avalable() else 'cpu'

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(20, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32 ,1),
#             # nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.network(x)

# # =================== Data preprocessing =====================
# X, y = make_classification(
#     n_samples = 1000, n_classes = 2,
#     n_features = 20, n_informative = 15, n_redundant = 5, random_state = 42
# )

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size = 0.2, stratify = y, random_state = 42
# )

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# X_train_t = torch.FloatTensor(X_train)
# # X_train_t = torch.tensor(X_train, dtype = torch.float32, device = device)
# # X_train_t = torch.tensor(X_train, dtype = torch.float32)
# X_test_t = torch.FloatTensor(X_test)
# y_train_t = torch.FloatTensor(y_train).view(-1,1)
# # y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
# y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# # Compare between different batch sizes
# batch_sizes = [8, 32, 128, 512]
# crit = nn.BCEWithLogitsLoss()

# batch_size_results = []
# for bs in batch_sizes:
#     dataset = TensorDataset(X_train_t, y_train_t)
#     dataloader = DataLoader(dataset, batch_size = bs, shuffle = True)

#     losses = []
#     num_epochs = 5
#     model = SimpleNet()
#     optimizer = optim.Adam(model.parameters(), lr = 1e-3)
#     # sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 3e-3, steps_per_epoch = len(dataloader), epochs = num_epochs)
#     for epoch in range(num_epochs):

#         epoch_loss = 0
#         for batch_x, batch_y in dataloader:
#             model.train()
#             optimizer.zero_grad()

#             output = model(batch_x)
#             loss = crit(output, batch_y)

#             loss.backward()
#             optimizer.step()
#             # sched.step()

#             epoch_loss += loss.item()           # avg by batch size
#         avg_loss = epoch_loss / len(dataloader) # avg by batch number
#         losses.append(avg_loss)

#     batch_size_results.append((bs, losses))
#     print(len(losses))
#     print(f"최종 손실 : {losses[-1]:.4f}")


# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# for batch_size, losses in batch_size_results:
#     plt.plot(losses, label=f'Batch={batch_size}', linewidth=2)

# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Training Loss', fontsize=12)
# plt.title('Effect of Batch Size', fontsize=14, weight='bold')
# plt.legend(fontsize=10)
# plt.grid(alpha=0.3)

# plt.subplot(1, 2, 2)
# final_losses = [losses[-1] for _, losses in batch_size_results]
# colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
# bars = plt.bar([str(bs) for bs in batch_sizes], final_losses,
#                color=colors, edgecolor='black', alpha=0.7)

# plt.xlabel('Batch Size', fontsize=12)
# plt.ylabel('Final Loss', fontsize=12)
# plt.title('Final Loss by Batch Size', fontsize=14, weight='bold')
# plt.grid(axis='y', alpha=0.3)

# # 값 표시
# for bar, loss in zip(bars, final_losses):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,                    # bar 중앙 정렬 + 높이
#              f'{loss:.4f}', ha='center', va='bottom', fontsize=10)        # 가로 : 중앙 정렬, 세로 : 바닥 정렬


# plt.tight_layout()
# print("배치 사이즈가 커지면 배치 개수가 적아지므로 같은 epoch 동안 파라미터 업데이트 횟수 감소")
# print()
# plt.show()
# plt.savefig('batch_size_effect.png', dpi=150, bbox_inches='tight')
# print("\n저장: batch_size_effect.png")
# plt.close()


# # =================== Effect by batch normalization =====================

# class NetWithoutBN(nn.Module):
#     """BatchNorm 없는 깊은 신경망"""
#     def __init__(self):
#         super(NetWithoutBN, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(20, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.network(x)

# class NetWithBN(nn.Module):
#     """BatchNorm 있는 깊은 신경망"""
#     def __init__(self):
#         super(NetWithBN, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(20, 64),
#             nn.BatchNorm1d(64),  # BatchNorm 추가, 매 변환 후 다음으로 넘길 때 항상 써야함
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.network(x)

# models_to_compare = [("Without BN", NetWithoutBN()),
#                      ("With NN",NetWithBN())]
# comparision_results = []

# X, y = make_classification(n_samples = 1000, n_classes = 2, n_features = 20,
#                            n_informative = 15, n_redundant = 5, random_state = 42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
# X_train_t = torch.FloatTensor(X_train)
# X_test_t = torch.FloatTensor(X_test)
# y_train_t = torch.FloatTensor(y_train).view(-1, 1)
# y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# batch_size = 32

# dataset = TensorDataset(X_train_t, y_train_t)
# dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
# crit = nn.BCELoss()


# for model_name, model in models_to_compare:
#     optimizer = optim.Adam(model.parameters(), lr = 1e-3)
#     losses = []
#     num_epochs = 5

#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for batch_X, batch_y in dataloader:
#             model.train()
#             optimizer.zero_grad()

#             output = model(batch_X)
#             loss = crit(output, batch_y)

#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#         avg_loss = epoch_loss / len(dataloader)
#         losses.append(avg_loss)

#         print(f"{epoch+1} : {avg_loss:.3f}")
#     comparision_results.append((model_name, losses))

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# for model_name, losses in comparision_results:
#     plt.plot(losses, label=model_name, linewidth=2)

# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Training Loss', fontsize=12)
# plt.title('BatchNorm Effect on Training', fontsize=14, weight='bold')
# plt.legend(fontsize=11)
# plt.grid(alpha=0.3)

# plt.subplot(1, 2, 2)
# for model_name, losses in comparision_results:
#     plt.plot(losses, label=model_name, linewidth=2)

# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Training Loss (log scale)', fontsize=12)
# plt.title('BatchNorm Effect (Log Scale)', fontsize=14, weight='bold')
# plt.yscale('log')
# plt.legend(fontsize=11)
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.show()
# plt.savefig('batchnorm_effect.png', dpi=150, bbox_inches='tight')
# print("\n저장: batchnorm_effect.png")
# plt.close()

# =================== Gradient stabilization by batch normalization =====================

X, y = make_classification(n_samples = 1000, n_classes = 2, n_features = 20,
                           n_informative = 15, n_redundant = 5, random_state = 42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

batch_size = 32

class NetWithoutBN(nn.Module):
    def __init__(self):
        super(NetWithoutBN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class NetWithBN(nn.Module):
    def __init__(self):
        super(NetWithBN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),  # BatchNorm 추가, 매 변환 후 다음으로 넘길 때 항상 써야함
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

crit = nn.BCELoss()
# gradient 수집
grad_hist_without_bn = []
grad_hist_with_bn = []

# 새 모델 생성
model_without_bn = NetWithoutBN()
model_with_bn = NetWithBN()

def create_hook_fn(gradient_list):
    def hook_fn(grad):
        gradient_list.append(grad.detach().abs().mean().item())
        return grad
    return hook_fn

for name, param in model_without_bn.named_parameters():
    if "weight" in name:
        param.register_hook(create_hook_fn(grad_hist_without_bn))

        
for name, param in model_with_bn.named_parameters():
    if "weight" in name:
        param.register_hook(create_hook_fn(grad_hist_with_bn))

# batch_X = X_train_t[:batch_size]   # batch size : 23
# batch_y = y_train_t[:batch_size]   # batch size : 23

# # without bn
# output_without = model_without_bn(batch_X)
# loss_without = crit(output_without, batch_y)
# loss_without.backward()


# # with bn
# output_with = model_with_bn(batch_X)
# loss_with = crit(output_with, batch_y)
# loss_with.backward()

# grad_hist_without_bn.reverse()
# grad_hist_with_bn.reverse()

# print("Without BN")
# for i, grad in enumerate(grad_hist_without_bn):
#     print(f' Layer {i+1}: {grad:.4f}')

# print("With BN")
# for i, grad in enumerate(grad_hist_with_bn):
#     print(f' Layer {i+1}: {grad:.4f}')

# # 시각화
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Without BN
# ax1 = axes[0]
# layers = list(range(1, len(grad_hist_without_bn) + 1))
# ax1.bar(layers, grad_hist_without_bn, color='#e74c3c',
#         edgecolor='black', alpha=0.7)
# ax1.set_xlabel('Layer Number', fontsize=11)
# ax1.set_ylabel('Gradient Magnitude', fontsize=11)
# ax1.set_title('Without BatchNorm', fontsize=12, weight='bold')
# ax1.set_xticks(layers)
# ax1.grid(axis='y', alpha=0.3)

# # With BN
# ax2 = axes[1]
# layers = list(range(1, len(grad_hist_with_bn) + 1))
# ax2.bar(layers, grad_hist_with_bn, color='#2ecc71',
#         edgecolor='black', alpha=0.7)
# ax2.set_xlabel('Layer Number', fontsize=11)
# ax2.set_ylabel('Gradient Magnitude', fontsize=11)
# ax2.set_title('With BatchNorm', fontsize=12, weight='bold')
# ax2.set_xticks(layers)
# ax2.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.show()
# plt.savefig('part5_gradient_stability.png', dpi=150, bbox_inches='tight')
# print("\n저장: part5_gradient_stability.png")
# plt.close()

# print("\n분석:")
# print("  - BatchNorm 없음: 층마다 그래디언트 크기 차이가 큼")
# print("  - BatchNorm 있음: 적어도 8개 층까지는 유사하게 그래디언트가 비슷한 크기")
# print("  - 결과: 더 안정적인 학습 가능")

# =================== Gradient stabilization by batch normalization with batch size =====================

batch_sizes_test = [8, 16, 32, 64]

results_comparison = {}

for batch_size in batch_sizes_test:
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # without bn
    model_no_bn = NetWithoutBN()
    optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=0.001)

    losses_no_bn = []

    for epoch in range(5):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            model_no_bn.train()
            optimizer_no_bn.zero_grad()

            output = model_no_bn(batch_X)
            loss = crit(output, batch_y)
            loss.backward()
            optimizer_no_bn.step()

            epoch_loss += loss.item()
        losses_no_bn.append(epoch_loss / len(dataloader))

    # with bn
    model_with_bn = NetWithBN()
    optimizer_with_bn = optim.Adam(model_with_bn.parameters(), lr=0.001)

    losses_with_bn = []

    for epoch in range(5):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            model_with_bn.train()
            optimizer_with_bn.zero_grad()

            output = model_with_bn(batch_X)
            loss = crit(output, batch_y)
            loss.backward()
            optimizer_with_bn.step()

            epoch_loss += loss.item()
        losses_with_bn.append(epoch_loss / len(dataloader))

    results_comparison[batch_size] = {
        'without_bn': losses_no_bn,
        'with_bn': losses_with_bn
    }

    print(f' without bn 최종 손실: {losses_no_bn[-1]:.4f}')
    print(f' with bn 최종 손실: {losses_with_bn[-1]:.4f}')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))


for idx, batch_size in enumerate(batch_sizes_test):
    ax = axes[idx // 2, idx % 2]

    losses_no_bn = results_comparison[batch_size]['without_bn']
    losses_with_bn = results_comparison[batch_size]['with_bn']

    ax.plot(losses_no_bn, label='Without BN', linewidth=2, color='#e74c3c')
    ax.plot(losses_with_bn, label='With BN', linewidth=2, color='#2ecc71')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title(f'Batch Size = {batch_size}', fontsize=12, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('part5_batchnorm_various_batch_sizes.png', dpi=150, bbox_inches='tight')
print("\n저장: part5_batchnorm_various_batch_sizes.png")
plt.close()
# 결론 : 배치 사이즈가 크면 클수록 배치 정규화를 할수록 학습이 잘 된다.