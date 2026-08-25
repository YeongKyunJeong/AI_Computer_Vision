# Transfer_learning_strategy.py

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


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

def show_sample_images(dataset, num_images = 5):
    fig, axes = plt.subplots(1, num_images, figsize = (15, 3))

    for i in range(num_images):
        img, label = dataset[i]

        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"{class_names[label]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# show_sample_images(dataset_train_small)

# train per epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0 
    correct = 0 
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        l1_norm = sum( p.abs().sum() for name, p in model.named_parameters()
                      if 'weight' in name)
        loss = criterion(outputs, labels)
        cost = loss + 1e-4*l1_norm

        cost.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item() # eq

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc

# evaluation
def evaluate_model(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    eval_loss = running_loss / total
    eval_acc = 100.0 * correct / total

    return eval_loss, eval_acc

# ============================== Partial Fine Tuning ==============================
learning_rate = 0.001
num_epochs = 10
criterion = nn.CrossEntropyLoss()

model_partial = models.resnet18(pretrained = True)

for p in model_partial.parameters():
    p.requires_grad = False

for p in model_partial.layer4.parameters():
    p.requires_grad = True

model_partial.fc =  nn.Linear(model_partial.fc.in_features, num_classes)

model_partial = model_partial.to(device)

optimizer_partial = optim.Adam(
    [
        {'params' : model_partial.layer4.parameters()},
        {'params' : model_partial.fc.parameters()}
    ],
    lr = learning_rate
)

# print(f"Trainable parameters : {sum(p.numel() for p in model_partial.parameters() if p.requires_grad):,}")

history_partial = {"train_loss": [], "train_acc": [], "test_acc": []}
print('\nTraining starts...')
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model_partial, train_dataloader, 
                                            criterion, optimizer_partial, device)

    _, test_acc = evaluate_model(model_partial, test_dataloader, criterion, device)

    history_partial['train_loss'].append(train_loss)
    history_partial['train_acc'].append(train_acc)
    history_partial['test_acc'].append(test_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] -"
          f"Train Loss : [{train_loss:.4f}], "
          f"Train Acc : [{train_acc:.2f}], "
          f"Test Acc : [{test_acc:.2f}], "
          )

elapsed_time_partial = time.time() - start_time

print(f"Total training time: {elapsed_time_partial:.2f} s")
print(f"Final test accuracy: {history_partial["test_acc"][-1]:.2f} %")


# ============================== Full Fine Tuning ==============================

model_full = models.resnet18(pretrained=True)

for param in model_full.parameters():
    param.requires_grad = True

num_features = model_full.fc.in_features
model_full.fc = nn.Linear(num_features, num_classes)

model_full = model_full.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_full = optim.Adam(model_full.fc.parameters(), lr=learning_rate)

# print(f"Trainable parameters : {sum(p.numel() for p in model_full.parameters() if p.requires_grad):,}")

history_full = {'train_loss': [], 'train_acc': [], 'test_acc': []}
print('\nTraining starts...')
start_time = time.time()


for epoch in range(num_epochs):

    train_loss, train_acc = train_one_epoch(model_full, train_dataloader,
                                           criterion, optimizer_full, device)

    _, test_acc = evaluate_model(model_full, test_dataloader, criterion, device)

    history_full['train_loss'].append(train_loss)
    history_full['train_acc'].append(train_acc)
    history_full['test_acc'].append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_acc:.2f}%, '
          f'Test Acc: {test_acc:.2f}%')

elapsed_time_full = time.time() - start_time
print(f"Total training time: {elapsed_time_full:.2f} s")
print(f"Final test accuracy: {history_full["test_acc"][-1]:.2f} %")

# ============================== Partial vs Full Fine Tuning ==============================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

epochs = range(1, num_epochs + 1)

# 1. Training acc
axes[0].plot(epochs, history_partial['train_acc'], 'g-s', label='Partial Fine-tune', linewidth=2)
axes[0].plot(epochs, history_full['train_acc'], 'r-^', label='Full Fine-tune', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Train Accuracy (%)', fontsize=12)
axes[0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 2. Test acc
axes[1].plot(epochs, history_partial['test_acc'], 'g-s', label='Partial Fine-tune', linewidth=2)
axes[1].plot(epochs, history_full['test_acc'], 'r-^', label='Full Fine-tune', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[1].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Result Summary
print('='*80)

results = [
    ['Partial Fine-tune', history_partial['test_acc'][-1], elapsed_time_partial,
     history_partial['train_acc'][-1] - history_partial['test_acc'][-1]],
    ['Full Fine-tune', history_full['test_acc'][-1], elapsed_time_full,
     history_full['train_acc'][-1] - history_full['test_acc'][-1]]
]

print(f'{"Strategy":<20} {"Test Acc (%)":<15} {"Time (sec)":<15} {"Overfit Gap (%)":<15}')
print('-'*80)
for result in results:
    print(f'{result[0]:<20} {result[1]:<15.2f} {result[2]:<15.2f} {result[3]:<15.2f}')
    
# print('\nInsight::')
# print('  - Overfit Gap = Train Acc - Test Acc (과적합 정도)')
# print('  - 작은 데이터셋에서는 Partial Fine-tuning이 좋은 균형점을 제공')
# print('  - Full Fine-tuning은 과적합 위험이 있으므로 주의가 필요')

# print('\n  - epoch 수가 적고 early stopping을 적용하지 않아 overfitting gap이 발생할 가능성이 큼')