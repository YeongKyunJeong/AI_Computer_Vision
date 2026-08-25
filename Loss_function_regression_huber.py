# Loss_function_regression_huber.py
import torch
from torch import nn
import matplotlib.pyplot as plt

delta_values = [0.5, 1.0, 2.0, 5.0]

# 오차 범위
errors = torch.linspace(-10, 10, 200)

plt.figure(figsize=(12, 5))

# Huber Loss 계산
for delta in delta_values:
    huber = nn.HuberLoss(delta=delta, reduction='none')
    # reduction : 출력된 huber loss를 어떻게 값으로 반환할지
    # 예 : 'mean'이면 행별 huber loss를 평균내서 반환함
    # 일반적으로 reduction='mean'이나 평균을 반환하기 때문에 그래프에서는 사용 안 함
    # 0을 실제값으로, errors를 예측값으로 설정
    loss_values = huber(errors, torch.zeros_like(errors))
    plt.plot(errors.numpy(), loss_values.numpy(),
             label=f'delta={delta}', linewidth=2)

# MSE와 MAE 비교
mse_values = 0.5 * errors**2
mae_values = torch.abs(errors)

plt.plot(errors.numpy(), mse_values.numpy(),
         'k--', linewidth=2, label='MSE', alpha=0.5)
plt.plot(errors.numpy(), mae_values.numpy(),
         'k:', linewidth=2, label='MAE', alpha=0.5)

plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.title('Huber Loss with Different Delta Values', fontsize=14, weight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xlim(-10, 10)
plt.ylim(0, 50)

plt.tight_layout()
plt.show()
plt.savefig('part1_huber_delta_effect.png', dpi=150, bbox_inches='tight')
print("\n저장: part1_huber_delta_effect.png")
plt.close()

print("\n분석:")
print("  delta가 작을수록 MAE에 가까움 (큰 오차에 관대)")
print("  error가 클수록 MAE에 가까움")
print("  delta가 클수록 MSE에 가까움 (작은 오차에 민감)")
print("  error가 작을수록 MSE에 가까움")
print("  delta=1.0이 일반적으로 좋은 기본값")