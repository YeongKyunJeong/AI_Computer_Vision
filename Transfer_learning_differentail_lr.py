# Transfer_learning_differentail_lr.py

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from Transfer_learning_strategy import show_sample_images, train_one_epoch, evaluate_model


plt.rcParams["font.family"] = "Nanum Gothic"
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale = (0.6, 1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.485, 0.456, 0.406),
                         std = (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406),
                         std = (0.229, 0.224, 0.225))
])

dataset_train_full = datasets.CIFAR10(
    root = "./data", train = True, download = True,
    transform = transform_train
)

dataset_test = datasets.CIFAR10(
    root = "./data", train = False, download = False,
    transform = transform_test
)

class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)

# small data senario

# 10 class, 50 samples per class
selected_indices = []
class_counts = {i : 0 for i in range(10)}
for idx, (image, label) in enumerate(dataset_train_full):
    if class_counts[label] < 50:
        selected_indices.append(idx)
        class_counts[label] += 1

    if len(selected_indices) >= 500:
        break

dataset_train_small = Subset(dataset_train_full, selected_indices)

batch_size = 32

train_dataloader =\
DataLoader(dataset_train_small, batch_size = batch_size,
           shuffle = True, num_workers = 2)

test_dataloader =\
DataLoader(dataset_test, batch_size = 2*batch_size,
           shuffle = False, num_workers = 2)

# print(len(train_dataloader))
# print(len(test_dataloader))


# ============================== Uniform lr ==============================
num_epochs = 10
criterion = nn.CrossEntropyLoss()

model_uniform_lr = models.resnet18(pretrained=True)

for param in model_uniform_lr.parameters():
    param.requires_grad = False
for param in model_uniform_lr.layer4.parameters():
    param.requires_grad = True

model_uniform_lr.fc = nn.Linear(model_uniform_lr.fc.in_features, num_classes)

model_uniform_lr = model_uniform_lr.to(device)

base_lr = 0.001

optimizer_uniform = optim.Adam(
    [{'params':model_uniform_lr.layer4.parameters()},
     {'params':model_uniform_lr.fc.parameters()}],
    lr=base_lr)

history_uniform_lr = {'train_loss': [], 'train_acc': [], 'test_acc': []}

print('\nTraining starts...')
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model_uniform_lr, train_dataloader,
                                           criterion, optimizer_uniform, device)
    _, test_acc = evaluate_model(model_uniform_lr, test_dataloader, criterion, device)

    history_uniform_lr['train_loss'].append(train_loss)
    history_uniform_lr['train_acc'].append(train_acc)
    history_uniform_lr['test_acc'].append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train acc: {train_acc:.2f}%, Test acc: {test_acc:.2f}%')

print(f'\nFinal test acc: {history_uniform_lr["test_acc"][-1]:.2f}%')


# ============================== Differentail lr ==============================

model_diff_lr = models.resnet18(pretrained=True)

for param in model_diff_lr.parameters():
    param.requires_grad = False
for param in model_diff_lr.layer4.parameters():
    param.requires_grad = True

model_diff_lr.fc = nn.Linear(model_diff_lr.fc.in_features, num_classes)

model_diff_lr = model_diff_lr.to(device)

backbone_lr = 0.001
head_lr = 0.01       # backbone_lr * 10

optimizer_diff = optim.Adam(
    [{'params':model_diff_lr.layer4.parameters(), 'lr': backbone_lr}, # backbone
     {'params':model_diff_lr.fc.parameters(), 'lr': head_lr}],        # head(clf)
 )

history_diff_lr = {'train_loss': [], 'train_acc': [], 'test_acc': []}

print('\nTraining starts...')
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model_diff_lr, train_dataloader,
                                           criterion, optimizer_diff, device)
    _, test_acc = evaluate_model(model_diff_lr, test_dataloader, criterion, device)

    history_diff_lr['train_loss'].append(train_loss)
    history_diff_lr['train_acc'].append(train_acc)
    history_diff_lr['test_acc'].append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train acc: {train_acc:.2f}%, Test acc: {test_acc:.2f}%')

print(f'\nFinal test acc: {history_diff_lr["test_acc"][-1]:.2f}%')


# ============================== Uniform vs Differential lr ==============================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

epochs = range(1, num_epochs + 1)

axes[0].plot(epochs, history_uniform_lr['train_loss'], 'b-o', label='Uniform LR', linewidth=2)
axes[0].plot(epochs, history_diff_lr['train_loss'], 'r-s', label='Differential LR', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Train Loss', fontsize=12)
axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 2. 테스트 정확도 비교
axes[1].plot(epochs, history_uniform_lr['test_acc'], 'b-o', label='Uniform LR', linewidth=2)
axes[1].plot(epochs, history_diff_lr['test_acc'], 'r-s', label='Differential LR', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[1].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Result Summary
print('='*80)

print(f'Uniform LR final acc: {history_uniform_lr["test_acc"][-1]:.2f}%')
print(f'Differential LR final acc: {history_diff_lr["test_acc"][-1]:.2f}%')
print(f'Acc diff: {history_diff_lr["test_acc"][-1] - history_uniform_lr["test_acc"][-1]:.2f}%p')

# print('\nInsight:')
# print('  - 헤드(새 분류기)는 높은 LR로 빠르게 학습')
# print('  - 백본(사전학습 층)은 낮은 LR로 신중하게 조정')
# print('  - 이 전략은 catastrophic forgetting을 방지합니다')