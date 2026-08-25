#Loss_function_regression.py

import numpy as np
import torch
from torch import nn, optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# # ================= comparison sensitivity between loss function ==================
# mse_loss = nn.MSELoss()
# mae_loss = nn.L1Loss()
# huber_loss = nn.HuberLoss(delta = 0.1)

# y_true_with_outlier = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
# y_pred_normal = torch.tensor([11.0, 19.0, 31.0, 39.0, 51.0])  # 정상 예측
# y_pred_with_outlier = torch.tensor([11.0, 19.0, 31.0, 100.0, 51.0])  # 이상치 포함

# print("\n정상 예측 (오차 모두 작음):")
# print("실제:", y_true_with_outlier.numpy())
# print("예측:", y_pred_normal.numpy())

# mse_normal = mse_loss(y_pred_normal, y_true_with_outlier)
# mae_normal = mae_loss(y_pred_normal, y_true_with_outlier)
# huber_normal = huber_loss(y_pred_normal, y_true_with_outlier)

# print(f"\nMSE:   {mse_normal.item():.4f}")
# print(f"MAE:   {mae_normal.item():.4f}")
# print(f"Huber: {huber_normal.item():.4f}")

# print("\n이상치 포함 예측 (하나의 큰 오차):")
# print("실제:", y_true_with_outlier.numpy())
# print("예측:", y_pred_with_outlier.numpy())
# print("오차:", (y_pred_with_outlier - y_true_with_outlier).numpy())

# mse_outlier = mse_loss(y_pred_with_outlier, y_true_with_outlier)
# mae_outlier = mae_loss(y_pred_with_outlier, y_true_with_outlier)
# huber_outlier = huber_loss(y_pred_with_outlier, y_true_with_outlier)

# print(f"\nMSE:   {mse_outlier.item():.4f} (증가율: {mse_outlier/mse_normal:.1f}배)")
# print(f"MAE:   {mae_outlier.item():.4f} (증가율: {mae_outlier/mae_normal:.1f}배)")
# print(f"Huber: {huber_outlier.item():.4f} (증가율: {huber_outlier/huber_normal:.1f}배)")

# print("\n분석:")
# print("  MSE는 이상치에 매우 민감 (제곱 때문)")
# print("  MAE는 이상치에 강건 (절댓값만 고려)")
# print("  Huber는 중간 (작은 오차는 제곱, 큰 오차는 선형)")

# ================= data preprocessing ==================
X, y = make_regression(n_samples = 500, n_features = 10, noise = 10., random_state = 42)
# noise : std of gaussian noise applied to output

print(y[:10])

# 이상치 추가
n_outlier = int(0.1*len(y))
outlier_indices = np.random.choice(len(y), n_outlier, replace = False)
y[outlier_indices] += np.random.randn(n_outlier)*50

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        count = 0
        for m in self.network:
            if isinstance(m, nn.Linear):
                if count == 2:
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    count += 1
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

loss_functions = {
    "MSE" : nn.MSELoss(),
    "MAE" : nn.L1Loss(),
    "Huber" : nn.HuberLoss(delta = 1.0)
}

results = {}

for loss_name, crit in loss_functions.items():
    # 모델 초기화
    model = RegressionModel()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    # sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 3e-2, steps_per_epoch = y_train_t.size(0), epochs = 100)

    # 학습
    train_losses = []
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train() # 학습 모드
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = crit(output, y_train_t)

        loss.backward()
        optimizer.step()
        # sched.step()

        train_losses.append(loss.item())

    model.eval()  # 훈련에 사용되는 모듈을 비활성화
    with torch.no_grad():
        test_pred = model(X_test_t)

        # 모든 지표로 평가
        test_mse = nn.MSELoss()(test_pred, y_test_t).item()
        test_mae = nn.L1Loss()(test_pred, y_test_t).item()
        test_huber = nn.HuberLoss()(test_pred, y_test_t).item()

    results[loss_name] = {
        "train_losses" : train_losses,
        "test_mse" : test_mse,
        "test_mae" : test_mae,
        "test_huber" : test_huber,
        "predictions" : test_pred
    }

    print(f"최종 훈련 소실 : {train_losses[-1]:.4f}" )
    print(f"Test MSE: {test_mse:.4f}" )
    print(f"Test MAE : {test_mae:.4f}" )
    print(f"Test_Huber : {test_huber:.4f}" )
    print("-"*30)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 학습 곡선 비교
ax1 = axes[0, 0]
colors = ['#e74c3c', '#3498db', '#2ecc71']
for (loss_name, result), color in zip(results.items(), colors):     # zip 함수 사용 확인
                                                                    # results.items() => (loss_name, result)
    ax1.plot(result['train_losses'], label=loss_name, linewidth=2, color=color)

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Training Loss', fontsize=11)
ax1.set_title('Training Loss Comparison', fontsize=12, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 2. 테스트 성능 비교 (MAE 기준)
ax2 = axes[0, 1]
loss_names = list(results.keys())
test_maes = [results[name]['test_mae'] for name in loss_names]

bars = ax2.bar(loss_names, test_maes, color=colors, edgecolor='black', alpha=0.7)
ax2.set_ylabel('Test MAE', fontsize=11)
ax2.set_title('Test MAE Comparison', fontsize=12, weight='bold')
ax2.grid(axis='y', alpha=0.3)

# 값 표시
for bar, mae in zip(bars, test_maes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{mae:.2f}', ha='center', va='bottom', fontsize=10, weight='bold') # 글자 정렬

# 3. 예측 vs 실제 (MAE 모델)
ax3 = axes[1, 0]
mae_predictions = results['MAE']['predictions'].numpy().flatten()
ax3.scatter(y_test, mae_predictions, alpha=0.6, edgecolors='black', linewidths=0.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],         # 값 범주 설정
         'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('True Values', fontsize=11)
ax3.set_ylabel('Predicted Values', fontsize=11)
ax3.set_title('MAE Model: Prediction vs True', fontsize=12, weight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. 오차 분포 비교
ax4 = axes[1, 1]
for loss_name, color in zip(loss_names, colors):
    predictions = results[loss_name]['predictions'].numpy().flatten()
    errors = y_test - predictions
    ax4.hist(errors, bins=30, alpha=0.5, label=loss_name, color=color)

ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error') # x축 vertical line 그리기
ax4.set_xlabel('Prediction Error', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Error Distribution', fontsize=12, weight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('part1_regression_loss_comparison.png', dpi=150, bbox_inches='tight')
print("\n저장: part1_regression_loss_comparison.png")
plt.close()

