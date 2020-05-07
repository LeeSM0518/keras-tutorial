# CHAPTER 07. 시계열수치입력 수치 예측 모델 레시피

각 모델에 코사인(cosine) 데이터를 학습시킨 후, 처음 일부 데이터를 알려주면 이후 코사인 형태의 데이터를 얼마나 잘 예측하는지 테스트해보자.

<br>

## 1. 데이터셋 준비

먼저 코사인 데이터를 만들어보자.

```python
import numpy as np

signal_data = np.cos(np.arange(1600) * (20 * np.pi / 1000))[:, None]
```

> 시간의 흐름에 따라 진폭이 -1.0 에서 1.0 사이로 변하는 1,600 개의 실수값을 생성한다.

<br>

생성한 데이터를 확인해보자.

```python
%matplotlib inline
import matplotlib.pyplot as plt

plot_x = np.arange(1600)
plot_y = signal_data
plt.plot(plot_x, plot_y)
plt.show()
```

![image](https://user-images.githubusercontent.com/43431081/81267427-6d784180-9081-11ea-8f76-7457a9879f21.png)

<br>



