# CHAPTER 03. 수치입력 다중클래스분류모델 레시피

수치를 입력해서 다중클래스를 분류할 수 있는 모델들에 대하여 알아보자.

<br>

## 1. 데이터셋 준비

훈련에 사용할 임의의 값을 가진 <u>인자 12개로</u> 구성된 <u>입력(x) 1000개와</u> 각 입력에 대해 <u>0에서 9까지 10개의 값 중 임의로 지정된 출력(y)을</u> 가지는 데이터셋을 생성해보자. 데이터는 100개를 준비해보자.

```python
import numpy as np

# 데이터셋 생성
x_train = np.random.random((1000, 12))
y_train = np.random.randint(10, size=(1000, 1))
x_test = np.random.random((100, 12))
y_test = np.random.randint(10, size=(100, 1))
```

> 데이터셋의 12개 인자(x) 및 라벨값(y) 모두 무작위 수이다.

<br>

12개 입력 인자 중 첫 번째와 두 번째 인자 값만 이용하여 2차원으로 데이터 분포를 살펴보자. 라벨값에 따라 점의 색상을 다르게 표시했다.

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

![image](https://user-images.githubusercontent.com/43431081/81095432-f38c6f00-8f3f-11ea-8723-755a6194f1e9.png)

<br>

이번에는 첫 번째, 두 번째, 세 번째의 인자값을 이용하여 3차원으로 그래프를 확인해보자.

```python
# 데이터셋 확인 (3차원)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_x = x_train[:, 0]
plot_y = x_train[:, 1]
plot_z = x_train[:, 2]
plot_color = y_train.reshape(1000,)

ax.scatter(plot_x, plot_y, plot_z, c=plot_color)
plt.show()
```

![image](https://user-images.githubusercontent.com/43431081/81095650-4bc37100-8f40-11ea-8583-ccdc7a9cb8c2.png)

<br>

## 2. 데이터셋 전처리

다중클래스분류인 경우에는 클래스별로 확률값을 지정하기 위해서는 **"one-hot 인코딩"을** 사용한다.

one-hot 인코딩은 아래 코드와 같이 케라스에서 제공하는 **"to_categorical()"** 함수로 쉽게 처리할 수 있다.

```python
y_train = np.random.randint(10, size=(1000, 1))
y_train = to_categorical(y_train, num_classes=10) # one-hot 인코딩

y_test = np.random.randint(10, size=(100, 1))
y_test = to_categorical(y_test, num_classes=10) # one-hot 인코딩
```

<br>

## 3. 레이어 준비

새롭게 소개되는 블록은 **'softmax'** 이다.

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
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_softmax_s.png" alt="img"></td>
      <td style="text-align: center">softmax</td>
      <td style="text-align: left">활성화 함수로 입력되는 값을 클래스별로 확률 값이 나오도록 출력시킵니다. 이 확률값을 모두 더하면 1이 됩니다. 다중클래스 모델의 출력층에 주로 사용되며, 확률값이 가장 높은 클래스가 모델이 분류한 클래스입니다.</td>
    </tr>
  </tbody>
</table>

<br>

## 4. 모델 준비

### 퍼셉트론 신경망 모델

Dense 레이어가 하나이고 뉴런의 수도 하나인 가장 기본적인 퍼셉트론 신경망 모델이다. 즉, 웨이트(w) 하나, 바이어스(b) 하나로 전형적인 `Y = w * X + b` 를 풀기 위한 모델이다.

다중클래스 분류이므로 **출력 레이어는 softmax 활성화 함수를** 사용했다.

```python
model = Sequential()
model.add(Dense(10, input_dim=12, activation='softmax'))
```

<img src="http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_1m.png">

<br>

### 다층퍼셉트론 신경망 모델

Dense 레이어가 두 개인 다층퍼셉트론 신경망 모델이다. 첫 번째 레이어는 64개의 뉴런을 가진 Dense 레이어이고 오류 역전파가 용이한 relu 활성화 함수를 사용한다.

출력 레이어인 두 번째 레이어는 클래스별 확률값을 출력하기 위해 10개의 뉴런과 softmax 활성화 함수를 사용했다.

```python
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

<img src="http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_2m.png">

<br>

### 깊은 다층퍼셉트론 신경망 모델

Dense 레이어가 총 세 개인 다층퍼셉트론 신경망 모델이다. 첫 번째, 두 번째 레이어는 64개의 뉴런을 가진 Dense 레이어이고 오류 역전파가 용이한 relu 활성화 함수를 사용했다. 출력 레이어인 세 번째 레이어는 클래스별 확률값을 출력하기 위해 10개의 뉴런과 softmax 활성화 함수를 사용했다.

```python
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

<img src="http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_3m.png">

<br>

## 5. 전체 소스

### 퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import random

# 1. 데이터셋 생성하기
x_train = np.random.random((1000, 12))
y_train = np.random.randint(10, size=(1000, 1))

x_test = np.random.random((100, 12))
y_test = np.random.randint(10, size=(100, 1))

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(10, input_dim=12, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 3.0])
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

**실행결과**

```
100/100 [==============================] - 0s 248us/step
loss_and_metrics : [2.325002784729004, 0.10999999940395355]
```

<br>

### 다층퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import random

# 1. 데이터셋 준비하기
x_train = np.random.random((1000, 12))
y_train = np.random.randint(10, size=(1000, 1))
y_train = to_categorical(y_train, num_classes=10) # one-hot 인코딩
x_test = np.random.random((100, 12))
y_test = np.random.randint(10, size=(100, 1))
y_test = to_categorical(y_test, num_classes=10)   # one-hot 인코딩

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

# 5. 학습과정 확인하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 3.0])
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
100/100 [==============================] - 0s 236us/step
loss_and_metrics : [3.1299439239501954, 0.07000000029802322]
```

<br>

### 깊은 다층퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import random

# 1. 데이터셋 준비하기
x_train = np.random.random((1000, 12))
y_train = np.random.randint(10, size=(1000, 1))
y_train = to_categorical(y_train, num_classes=10) # one-hot 인코딩
x_test = np.random.random((100, 12))
y_test = np.random.randint(10, size=(100, 1))
y_test = to_categorical(y_test, num_classes=10)   # one-hot 인코딩

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 3.0])
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
100/100 [==============================] - 0s 275us/step
loss_and_metrics: [8.00372917175293, 0.10000000149011612]
```

<br>

## 6. 학습결과 비교

<table>
  <thead>
    <tr>
      <th style="text-align: center">퍼셉트론 신경망 모델</th>
      <th style="text-align: center">다층퍼셉트론 신경망 모델</th>
      <th style="text-align: center">깊은 다층퍼셉트론 신경망 모델</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_output_18_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_output_20_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_output_22_1.png" alt="img"></td>
    </tr>
  </tbody>
</table>