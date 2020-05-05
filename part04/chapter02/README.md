# CHAPTER 02. 수치입력 이진분류모델 레시피

이진분류를 위한 데이터셋 생성을 해보고 가장 간단한 퍼셉트론 신경망 모델부터 깊은 다층퍼셉트론 신경망 모델까지 구성 및 학습을 시켜보자.

<br>

## 1. 데이터셋 준비

훈련에 사용할 임의의 값을 가진 인자 12개로 구성된 입력(x) 1000 개와 각 입력에 대해 0과 1중 임의로 지정된 출력(y)을 가지는 데이터셋을 생성했다.

```python
import numpy as np

# 데이터셋 생성
x_train = np.random.random((1000, 12))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 12))
y_test = np.random.randint(2, size=(100, 1))
```

* 데이터셋의 12개 인자(x) 및 라벨값(y)은 모두 무작위 수이다.

<br>

12개 입력 인자 중 첫 번째와 두 번째 인자 값만 이용하여 2차원으로 데이터 분포를 살펴보자.

```python
%matplotlib inline
import matplotlib.pyplot as plt

# 데이터셋 확인 (2차원)
plot_x = x_train[:, 0]
plot_y = x_train[:, 1]
plot_color = y_train.reshape(1000,)

plt.scatter(plot_x, plot_y, c=plot_color)
plt.show()
```

![image](https://user-images.githubusercontent.com/43431081/81070870-d72c0a80-8f1e-11ea-82a6-b996180f94b2.png)

<br>

## 2. 레이어 준비

<table>
  <thead>
    <tr>
      <th style="text-align: center">블록</th>
      <th style="text-align: center">이름</th>
      <th style="text-align: left">설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_sigmoid_s.png" alt="img"></td>
      <td style="text-align: center">sigmoid</td>
      <td style="text-align: left">활성화 함수로 입력되는 값을 0과 1사이의 값으로 출력시킵니다. 출력값이 특정 임계값(예를 들어 0.5) 이상이면 양성, 이하이면 음성이라고 판별할 수 있기 때문에 이진분류 모델의 출력층에 주로 사용됩니다.</td>
    </tr>
  </tbody>
</table>

<br>

## 3. 모델 준비

### 퍼셉트론 신경망 모델

Dense 레이어가 하나이고, 뉴런의 수도 하나인 가장 기본적인 퍼셉트론 신경망 모델이다.

즉, 웨이트(w) 하나, 바이어스(b) 하나로 전형적인 `Y = w * X + b` 를 풀기 위한 모델이다. 이진분류이므로 출력 레이어는 **sigmoid 활성화 함수를** 사용한다.

```python
model = Sequential()
model.add(Dense(1, input_dim=12, activation='sigmoid'))
```

또는 활성화 함수를 블록을 쌓듯이 별로 레이어로 구성하여도 동일한 모델이다.

```python
model = Sequential()
model.add(Dense(1, input_dim=12))
model.add(Activation('sigmoid'))
```

<br>

### 다층퍼셉트론 신경망 모델

Dense 레이어가 두 개인 다층퍼셉트론 신경망 모델이다. 첫 번째 레이어는 64개의 뉴런을 가진 Dense 레이어이고 오류 역전파가 용이한 relu 활성화 함수를 사용했다.

출력 레이어인 두 번째 레이어는 0과 1사이의 값 하나를 출력하기 위해 1개의 뉴런과 sigmoid 활성화 함수를 사용했다.

```python
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

<img src="http://tykimos.github.io/warehouse/2017-8-18-Numerical_Input_Binary_Classification_Model_Recipe_2m.png">

<br>

### 깊은 다층퍼셉트론 신경망 모델

Dense 레이어가 총 세 개인 다층퍼셉트론 신경망 모델이다. 첫 번째, 두 번째 레이어는 64개의 뉴런을 가진 Dense 레이어이고 오류 역전파가 용이한 relu 활성화 함수를 사용했다.

출력 레이어인 세 번째 레이어는 0과 1사이의 값 하나를 출력하기 위해 1개의 뉴런과 sigmoid 활성화 함수를 사용했다.

```python
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

<img src="http://tykimos.github.io/warehouse/2017-8-18-Numerical_Input_Binary_Classification_Model_Recipe_3m.png">

<br>

## 4. 전체 소스

### 퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random

# 1. 데이터셋 생성하기
x_train = np.random.random((1000, 12))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 12))
y_test = np.random.randint(2, size=(100, 1))

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(1, input_dim=12, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('loss_and_metrics :', loss_and_metrics)
```

**실행 결과**

```
Epoch 1/1000
1000/1000 [==============================] - 0s 287us/step - loss: 0.7785 - accuracy: 0.5070
...
Epoch 1000/1000
1000/1000 [==============================] - 0s 21us/step - loss: 0.6886 - accuracy: 0.5330
100/100 [==============================] - 0s 232us/step
loss_and_metrics : [0.7123481369018555, 0.44999998807907104]
```

<br>

### 다층퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random

# 1. 데이터셋 생성하기
x_train = np.random.random((1000, 12))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 12))
y_test = np.random.randint(2, size=(100, 1))

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('loss_and_metrics :', loss_and_metrics)
```

**실행 결과**

```
Epoch 1/1000
1000/1000 [==============================] - 0s 297us/step - loss: 0.6950 - accuracy: 0.5160
...
Epoch 1000/1000
1000/1000 [==============================] - 0s 25us/step - loss: 0.4288 - accuracy: 0.8230
100/100 [==============================] - 0s 210us/step
loss_and_metrics : [0.8493837594985962, 0.5799999833106995]
```

<br>

### 깊은 다층퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random

# 1. 데이터셋 생성하기
x_train = np.random.random((1000, 12))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 12))
y_test = np.random.randint(2, size=(100, 1))

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('loss_and_metrics:', loss_and_metrics)
```

**실행 결과**

```
100/100 [==============================] - 0s 207us/step
loss_and_metrics: [4.6029095109552145, 0.5699999928474426]
```

<br>

## 5. 학습결과 비교

<table>
  <thead>
    <tr>
      <th style="text-align: center">퍼셉트론</th>
      <th style="text-align: center">다층퍼셉트론</th>
      <th style="text-align: center">깊은 다층퍼셉트론</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-18-Numerical_Input_Binary_Classification_Model_Recipe_03.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-18-Numerical_Input_Binary_Classification_Model_Recipe_04.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-18-Numerical_Input_Binary_Classification_Model_Recipe_05.png" alt="img"></td>
    </tr>
  </tbody>
</table>