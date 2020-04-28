# CHAPTER 04. 학습 조기종료 시키기

## 조기종료 시키기

학습 조기종료를 위해서는 **'EarlyStopping()'** 이라는 함수를 사용하며, **더 이상의 개선의 여지가 없을 때 학습을 종료시키는 콜백 함수이다.**

먼저 fit() 함수에서 EarlyStopping() 콜백함수를 지정하는 방법은 다음과 같다.

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()
hist = model.fit(x_train, y_train, epochs=3000, batch_size=10, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

<br>

EarlyStopping 콜백함수에서 설정할 수 있는 인자는 다음과 같다.

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                             mode='auto')
```

* **monitor** : 관찰하고자 하는 항목이다.
* **min_delta** : 개선되고 있다고 판단하기 위한 최소 변화랑
* **patience** : 개선이 없다고 바로 종료하지 않고 개선이 없는 에포크를 얼마나 기다려 줄 것인가를 지정
* **verbose** : 얼마나 자세하게 정보를 표시할 것인가를 지정
* **mode** : 관찰 항목에 대해 개선이 없다고 판단하기 위한 기준을 지정한다.
  * auth: 관찰하는 이름에 따라 자동으로 지정한다.
  * min: 관찰하고 있는 항목이 감소되는 것을 멈출 때 종료
  * max: 관찰하고 있는 항목이 증가되는 것을 멈출 때 종료

<br>

### 조기종료 콜백함수를 적용한 코드

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(3)

# 1. 데이터셋 준비하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]

# 데이터셋 전처리
x_train = x_train.reshape(50000, 784).astype('float32') / 255.0
x_val = x_val.reshape(10000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# 훈련셋과 검증셋 고르기
train_rand_idxs = np.random.choice(50000, 700)
val_rand_idxs = np.random.choice(10000, 300)

x_train = x_train[train_rand_idxs]
y_train = y_train[train_rand_idxs]
x_val = x_val[val_rand_idxs]
y_val = y_val[val_rand_idxs]

# 라벨데이터 원핫인코딩 (one-hot encoding) 처리
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=2, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()
hist = model.fit(x_train, y_train, epochs=3000, batch_size=10, validation_data=(x_val, y_val),
                 callbacks=[early_stopping])

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='tarin acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuracy : ' + str(loss_and_metrics[1]))
```

**실행 결과**

![image](https://user-images.githubusercontent.com/43431081/80436118-56c34380-8939-11ea-8a10-db9413381e01.png)

**평가 결과**

```
10000/10000 [==============================] - 0s 20us/step

loss : 1.8301704263687133
accuracy : 0.30889999866485596
```

* val_loss 값이 감소되다가 증가되자마자 학습이 종료되었다. val_loss 특성상 증가/감소를 반복하므로 val_loss가 증가되는 시점에 바로 종료하지 않고 지속적으로 증가되는 시점에서 종료해보자.

  ```python
  from keras.callbacks import EarlyStopping
  # 증가가 되었더라도 20 에포크 동안은 종료시키지 않는다.
  early_stopping = EarlyStopping(patience = 20)
  hist = model.fit(x_train, y_train, epochs=3000, batch_size=10,
                   validation_data=(x_val, y_val), callbacks=[early_stopping])
  ```

<br>

### 코드 변경 후 실행

![image](https://user-images.githubusercontent.com/43431081/80436401-00a2d000-893a-11ea-842d-eedb7da01199.png)

즉, 과적합이 발생되거나 성급하게 학습을 조기종료한 모델보다 적절히 조기종료한 모델의 정확도가 높게 나왔다.

* **세 가지 모델을 비교한 표**

  | 구분   | 과적합 | 성급한 조기종료 | 적절한 조기종료 |
  | ------ | ------ | --------------- | --------------- |
  | 손실값 | 3.73   | 1.43            | 1.34            |
  | 정확도 | 0.44   | 0.44            | 0.53            |