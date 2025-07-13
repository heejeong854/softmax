import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# 모델 정의
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

def softmax(logits):
    exps = torch.exp(logits - torch.max(logits))
    return exps / torch.sum(exps)

st.title("🧠 Softmax 숫자 인식 시뮬레이터 (간단 버전)")

st.markdown("""
MNIST 테스트 이미지 중 하나를 선택하여 모델의 예측 결과를 확인해보세요.
""")

# 이미지 transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 테스트셋 로드
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 슬라이더로 이미지 선택
index = st.slider("이미지 번호 (0~9999)", 0, len(mnist_test)-1, 0)
image, label = mnist_test[index]

# 정규화 해제해서 시각화용 이미지 생성
image_for_display = image * 0.3081 + 0.1307
st.image(image_for_display.squeeze().numpy(), caption=f"Ground Truth: {label}", width=150)

# 모델 정의 및 추론
model = SimpleNN()
model.eval()

with torch.no_grad():
    logits = model(image.unsqueeze(0)).squeeze()
    probs = softmax(logits).numpy()
    logits_np = logits.numpy()

predicted = int(np.argmax(probs))
st.write(f"### ✅ 예측 결과: {predicted}")

# 로짓 그래프
st.subheader("📊 Logits (모델 출력값)")
fig1, ax1 = plt.subplots()
ax1.bar(range(10), logits_np, color='gray')
ax1.set_xticks(range(10))
ax1.set_xlabel("클래스 (0~9)")
ax1.set_ylabel("로짓 값")
st.pyplot(fig1)

# softmax 확률 분포 그래프
st.subheader("📈 Softmax 확률 분포")
fig2, ax2 = plt.subplots()
bars = ax2.bar(range(10), probs, color='skyblue')
bars[predicted].set_color('orange')
ax2.set_xticks(range(10))
ax2.set_xlabel("클래스 (0~9)")
ax2.set_ylabel("확률")
ax2.set_ylim(0, 1)
st.pyplot(fig2)
