# Imbalanced_data_preprocess.py

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from collections import Counter

class ImabalancedDataset(Dataset):
    def __init__(self, n_samples = 2000, n_features = 20, n_classes = 4,
                 imbalanced_ratio = [0.5, 0.3, 0.15, 0.05]):
        self.n_classes = n_classes
        samples_per_class = [int(n_samples * ratio) for ratio in imbalanced_ratio]

        X_list = []
        y_list = []

        for class_idx in range(n_classes):
            n = samples_per_class[class_idx]
            means = np.random.randn(n_features) * (0.5 + class_idx + 1) # 1.5, 2.5, 3.5, ...
            covs = np.eye(n_features) * (0.5 + class_idx * 0.2) # 0.5, 0.7, 0.9, ...
            X_class = np.random.multivariate_normal(mean = means, cov = covs, size = n)
            y_class = np.full(n, class_idx)

            X_list.append(X_class)
            y_list.append(y_class)

        self.X = torch.FloatTensor(np.vstack(X_list))
        self.y = torch.LongTensor(np.vstack(y_list))

        class_counts = Counter(self.y.numpy())
        print(f"전체 클래스 분포 : {dict(sorted(class_counts.items()))}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MultiClassifier(nn.Module):
    def __init__(self, input_dim = 20, hidden_dim = 64, n_classes = 4):
        super(MultiClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, n_classes),
        )

    def __forward__(self, x):
        return self.network(x)

# ======================== Weighed Loss ========================
    
def get_labels_from_loader(loader: DataLoader) -> list[int]:
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.numpy().tolist())
    return labels

def get_class_weights(train_loader : DataLoader, n_classes = int) -> float[float]:
    labels = get_labels_from_loader(train_loader)
    class_counts = Counter(labels)
    n_samples = len(labels)

    weights = torch.FloatTensor([
        n_samples / (n_classes * class_counts.get(i, 1))
        for i in range(n_classes)
    ])

    return weights    # ====> give this weight nn.CrossEntropy(weight = weights)

def create_weighted_sampler(train_loader: DataLoader) -> WeightedRandomSampler:
    # For oversampling
    labels = get_labels_from_loader(train_loader)
    class_counts = Counter(labels)

    sample_weights = [1.0 / class_counts.get(label, 1) for label in labels]
    sampler = WeightedRandomSampler(
        weights = sample_weights, num_samples = len(sample_weights),
        replacement = True
    )

    return sampler

# ======================== Oversampling ========================

def create_weighted_sampler(train_loader : DataLoader) -> WeightedRandomSampler:
    labels = get_labels_from_loader(train_loader)
    class_counts = Counter(labels)

    sample_weights = [1.0 / class_counts.get(label, 1) for label in range(labels)]
    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(labels),
        replacement = True    # match counts of classes by replacement (oversampling)
    )

# ======================== Augmentation 1: Make new data with noise ========================

class AugmentedWrapper(Dataset):
    def __init__(self, train_dataset : Dataset, augment_ratio : float = 0.5, noise_std : float = 0.15):
        self.dataset = train_dataset
        self.base_len = len(train_dataset)
        self.augment_len = int(self.base_len * augment_ratio)
        self.total_len = self.base_len + self.augment_len
        self.noise_std = noise_std

        def __len__(self):
            return self.total_len

        def __getitem__(self, idx):
            if idx < self.base_len:
                return self.dataset[idx]
            else:
                base_idx = idx % self.base_len
                x, y = self.dataset[base_idx]
                noise = torch.randn_like(x) * self.noise_std

            return x + noise, y

# ======================== Augmentation 2: Random Flipping and Erasing ========================
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        transforms.RandomErasing(p = 0.5, scale = (0.02, 0.33),
                                 ratio = (0.3, 3.3), inplace = False)
    ]
)
