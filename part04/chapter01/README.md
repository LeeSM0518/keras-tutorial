# CHAPTER 01. 수치입력 수치 예측 모델 레시피

수치를 입력해서 수치를 예측하는 모델들에 대하여 알아보자.

선형회귀를 위한 가장 간단한 퍼셉트론 신경망 모델부터 깊은 다층퍼셉트론 신경망 모델까지 구성 및 학습을 시켜보자.

<br>

## 1. 데이터셋 준비

입력 x에 대해 2를 곱해 두 배 값을 갖는 출력 y가 되도록 데이터셋을 생성해보자.

**선형회귀 모델을** 사용한다면 **Y = w * X + b** 일 때, w가 2에 가깝고, b가 0.16에 가깝게 되도록 학습시켜보자.

```python
import numpy as np

# 데이터셋 생성
x_train = np.random.random((1000, 1))
y_train = x_train * 2 + np.random.random((1000, 1)) / 3.0
x_test = np.random.random((100, 1))
y_test = x_test * 2 + np.random.random((100, 1)) / 3.0

# 데이터 셋 확인
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(x_train, y_train, 'ro')
plt.plot(x_test, y_test, 'bo')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

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
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dataset_Vector_s.png" alt="img"></td>
      <td style="text-align: center">Input data, Labels</td>
      <td style="text-align: left">1차원의 입력 데이터 및 라벨입니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dense_s.png" alt="img"></td>
      <td style="text-align: center">Dense</td>
      <td style="text-align: left">모든 입력 뉴런과 출력 뉴런을 연결하는 전결합층입니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_Relu_s.png" alt="img"></td>
      <td style="text-align: center">relu</td>
      <td style="text-align: left">활성화 함수로 주로 은닉층에 사용됩니다.</td>
    </tr>
  </tbody>
</table>

<br>

## 3. 모델 준비

선형회귀 모델, 퍼셉트론 신경망 모델, 다층퍼셉트론 신경망 모델, 깊은 다층퍼셉트론 신경망 모델을 준비해보자.

<br>

### 선형회귀 모델

가장 간단한 1차 선형회귀 모델로 수치 예측을 해보자.

`Y = w * x + b` 

* **x, y** : 우리가 만든 데이터셋
* **w, b** : 회귀분석을 통하여 구할 것

w, b 값은 분산, 공분산, 평균을 이용하여 쉽게 구할 수 있다.

```python
w = np.cov(X, Y, bias=1)[0, 1] / np.var(X)
b = np.average(Y) - w * np.average(X)
```

<br>

### 퍼셉트론 신경망 모델

Dense 레이어가 하나이고, 뉴런의 수도 하나인 가장 기본적인 퍼셉트론 신경망 모델이다.

즉, 웨이트(w) 하나, 바이어스(b) 하나로 전형적인 `Y = w * X + b` 를 풀기 위한 모델이다.

```python
model = Sequential()
model.add(Dense(1, input_dim=1))
```

<img src="http://tykimos.github.io/warehouse/2017-8-12-Numerical_Prediction_Model_Recipe_1m.png">

<br>

### 다층퍼셉트론 신경망 모델

Dense 레이어가 두 개인 다층퍼셉트론 신경망 모델이다. 첫 번째 레이어는 64개의 뉴런을 가진 Dense 레이어이고 오류 역전파가 용이한 relu 활성화 함수를 사용했다. 출력 레이어인 두 번째 레이어는 하나의 수치값을 예측하기 위해서 1개의 뉴런을 가지며, 별도의 활성화 함수를 사용하지 않는다.

```python
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(1))
```

<img src="http://tykimos.github.io/warehouse/2017-8-12-Numerical_Prediction_Model_Recipe_2m.png">

<br>

### 깊은 다층퍼셉트론 신경망 모델

Dense 레이어가 총 세 개인 다층퍼셉트론 신경망 모델이다. 첫 번째, 두 번째 레이어는 64개의 뉴런을 가진 Dense 레이어이고 오류 역전파가 용이한 relu 활성화 함수를 사용했다. 출력 레이어인 세 번째 레이어는 하나의 수치값을 예측을 하기 위해서 1개의 뉴런을 가지며, 별도의 활성화 함수를 사용하지 않는다.

```python
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
```

<img src="http://tykimos.github.io/warehouse/2017-8-12-Numerical_Prediction_Model_Recipe_3m.png">

<br>

## 4. 전체 소스

### 선형회귀 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from sklearn.metrics import mean_squared_error
import random

# 1. 데이터셋 생성하기
x_train = np.random.random((1000, 1))
print('x_train\n', x_train)
y_train = x_train * 2 + np.random.random((1000, 1)) / 3.0
x_test = np.random.random((100, 1))
y_test = x_test * 2 + np.random.random((100, 1)) / 3.0

x_train = x_train.reshape(1000,)
print('\nx_train 2\n', x_train)
y_train = y_train.reshape(1000,)
x_test = x_test.reshape(100,)
y_test = y_test.reshape(100,)

# 2. 모델 구성하기
w = np.cov(x_train, y_train, bias=1)[0, 1] / np.var(x_train)
b = np.average(y_train) - w * np.average(x_train)

print(w, b)

# 3. 모델 평가하기
y_predict = w * x_test + b
mse = mean_squared_error(y_test, y_predict)
print('mse : ' + str(mse))
```

**실행 결과**

```
mse : 0.010769815153675245
```

<br>

### 퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random

# 1. 데이터셋 생성하기
x_train = np.random.random((1000, 1))
y_train = x_train * 2 + np.random.random((1000, 1)) / 3.0
x_test = np.random.random((100, 1))
y_test = x_test * 2 + np.random.random((100, 1)) / 3.0

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='mse')

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=50, batch_size=64)
w, b = model.get_weights()
print(w, b)

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.ylim(0.0, 1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 6. 모델 평가하기
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ' + str(loss))
```

**실행 결과**

```
Epoch 1/50
1000/1000 [==============================] - 0s 245us/step - loss: 2.4639
...
Epoch 50/50
1000/1000 [==============================] - 0s 16us/step - loss: 0.3081
[[0.30182666]] [0.7876509]
100/100 [==============================] - 0s 242us/step
loss : 0.25431362390518186
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
x_train = np.random.random((1000, 1))
y_train = x_train * 2 + np.random.random((1000, 1)) / 3.0
x_test = np.random.random((100, 1))
y_test = x_test * 2 + np.random.random((100, 1)) / 3.0

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='mse')

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=50, batch_size=64)

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.ylim(0.0, 1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 6. 모델 평가하기
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss :', loss)
```

**실행 결과**

```
Epoch 1/50
1000/1000 [==============================] - 0s 237us/step - loss: 1.2237
...
Epoch 50/50
1000/1000 [==============================] - 0s 19us/step - loss: 0.0091
100/100 [==============================] - 0s 158us/step
loss : 0.008369513899087905
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
x_train = np.random.random((1000, 1))
y_train = x_train * 2 + np.random.random((1000, 1)) / 3.0
x_test = np.random.random((100, 1))
y_test = x_test * 2 + np.random.random((100, 1)) / 3.0

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='mse')

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=50, batch_size=64)

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.ylim(0.0, 1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 6. 모델 평가하기
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss :', loss)
```

**실행 결과**

```
Epoch 1/50
1000/1000 [==============================] - 0s 380us/step - loss: 1.0465
...
Epoch 50/50
1000/1000 [==============================] - 0s 31us/step - loss: 0.0098
100/100 [==============================] - 0s 214us/step
loss : 0.008553402405232191
```