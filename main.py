import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from sklearn.datasets import load_digits

# PyTorch 모델 (MNIST에 맞춰 훈련된 간단한 MLP)
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Softmax 함수
def softmax(logits):
    exps = torch.exp(logits - torch.max(logits))
    return exps / exps.sum()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST sample images
st.title("🔢 Softmax 기반 숫자 이미지 분류 시뮬레이터")

st.markdown("숫자 이미지를 입력하면, 모델이 로짓(logits)과 softmax 확률을 시각화합니다.")

# 이미지 선택
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
index = st.slider("테스트 이미지 번호", 0, len(mnist_test)-1, 0)

image, label = mnist_test[index]
st.image(image.squeeze().numpy(), caption=f"Ground Truth: {label}", width=150)

# 모델 로드 (사전 훈련된 가중치는 없으니 랜덤 init 상태, 학습된 모델 연결 가능)
model = SimpleNN()
model.eval()
with torch.no_grad():
    logits = model(image.unsqueeze(0)).squeeze()
    probs = softmax(logits).numpy()
    logits_np = logits.numpy()

# 예측 결과
predicted = np.argmax(probs)

st.write(f"### ✅ 예측 결과: {predicted} (정답: {label})")

# 로짓 그래프
st.subheader("📊 Logits (모델 출력값, softmax 이전)")
fig1, ax1 = plt.subplots()
ax1.bar(range(10), logits_np, color='gray')
ax1.set_xticks(range(10))
ax1.set_xlabel("클래스 (0~9)")
ax1.set_ylabel("로짓 값")
st.pyplot(fig1)

# softmax 확률 그래프
st.subheader("📈 Softmax 확률 분포")
fig2, ax2 = plt.subplots()
bars = ax2.bar(range(10), probs, color='skyblue')
bars[predicted].set_color('orange')
ax2.set_xticks(range(10))
ax2.set_xlabel("클래스 (0~9)")
ax2.set_ylabel("확률")
ax2.set_ylim(0, 1)
st.pyplot(fig2)
