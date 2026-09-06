# Transfer_learning.py

import matplotlib.pyplot as plt
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

# 10 class, 300 samples per class
selected_indices = []
class_counts = {i : 0 for i in range(10)}
for idx, (image, label) in enumerate(dataset_train_full):
    if class_counts[label] < 300:
        selected_indices.append(idx)
        class_counts[label] += 1

    if len(selected_indices) >= 3000:
        break

dataset_train_small = Subset(dataset_train_full, selected_indices)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog','frog','horse','ship','truck']
num_classes = len(class_names)

# Dataloader
train_loader = DataLoader(dataset_train_small,
                          batch_size = 64,
                          shuffle = True,
                          num_workers = 2,
                          pin_memory = True)

test_loader = DataLoader(dataset_test,
                         batch_szie = 128,
                         shuffle = False,
                         num_workers = 2,
                         pin_memory = True)

# Model building
def build_model(strategy):

    model = models.resnet18(weights = models.Resnet18_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if strategy == "freeze":
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "partial":
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "full":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError("지원하지 않는 전략 입력")

    model.to(device)
    trainable_param = sum( p.numel() for p in model.parameters() if p.requires_grad == True)
    total_param = sum(p.numel for p in model.parameters())

    print(f"Trainable parameters : {trainable_param:,} / {total_param:,} ") # 100,000

    return model

# Train and Evaluate loop
def train_and_evaluate(strategy):

    model = build_model(strategy)

    head_params = list(model.fc.parameters())
    backbone_params = [param for name, param in model.named_parameters()
                       if 'fc' not in name and param.requires_grad == True] # fc.weight, fc.bias

    param_groups = []

    # Apply differential learning rate
    if backbone_params: # if backbone parameter exists
        param_groups.append({'params' : backbone_params, 'lr' : 1e-4})

    if head_params: # if parameter of classifier exists
        param_groups.append({'params' : head_params, 'lr' : 1e-3})

    optimizer = optim.Adam(params = param_groups, weight_decay = 1e-4) # L2 regularization

    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0. # accumulated loss per epoch
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            l1_norm = sum( p.abs().sum() for name, p 
                          in model.named_parameters() 
                          if 'weight' in name ) # L1 regularization
            loss = criterion(outputs, labels) 
            cost = loss + 1e-3*l1_norm
            cost.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100*correct/total
        print(f"Epoch : [{epoch+1} / {num_epochs}] "
              f"Loss : {epoch_loss:.4f} "
              f"Train_acc : {epoch_acc:2f}%")

    model.eval()
    correct = 0
    total = 0
    running_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            running_loss += criterion(outputs, labels).item() * images.size(0)
            total += labels.size(0)

            # predicted = outputs.argmax(1)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    test_loss = running_loss / total
    print(f"Test_loss : {test_loss:.4f}"
          f"Test_acc : {100*test_accuracy:.2f} "
          )
    return test_loss, test_accuracy

_, acc_freeze = train_and_evaluate('freeze')
_, acc_partial = train_and_evaluate('partial')
_, acc_full = train_and_evaluate('full')

results = {
    'Freeze': acc_freeze,
    'Partial (layer4, fc)': acc_partial,
    'Full': acc_full
}

print(f'\n{"Strategy":<30} {"Test_acc":>15}')
print('='*70)

for strategy_name, accuracy in results.items():
    print(f'{strategy_name:<30} {accuracy*100:>14.2f}%')

best_strategy = max(results, key=results.get)
print(best_strategy )

strategies = ['Freeze\n', 'Partial\n(layer4, fc)', 'Full\n(전체 학습)']
accuracies = [acc_freeze * 100, acc_partial * 100, acc_full * 100]

# bat chart
plt.figure(figsize=(10, 6))
bars = plt.bar(strategies, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'],
               alpha=0.8, edgecolor='black', linewidth=1.5)

# acc on bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Transfer Learning Strategy Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, max(accuracies) * 1.15])
plt.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
