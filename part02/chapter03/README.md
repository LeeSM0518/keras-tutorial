# CHAPTER 03. 학습과정 살펴보기

fit() 함수를 사용하면 화면에 로그를 찍어준다. 이 로그에는 학습에 대한 수치가 나타나는데, 이 수치를 그래프로 보는 것이 편하므로 케라스에서 제공하는 히스토리 기능을 이용하는 방법, 텐서보드와 연동하여 보는 방법, 콜백함수를 직접 만들어서 사용하는 방법에 대해서 살펴보자.

<br>

## 1. 히스토리 기능 사용하기

케라스에서 학습시킬 때 **fit()** 함수를 사용한다. 이 함수의 반환 값으로 히스토리 객체를 얻을 수 있다.

* **히스토리 객체 정보**

  * 매 에포크 마다의 훈련 손실값 (loss)
  * 매 에포크 마다의 훈련 정확도 (acc)
  * 매 에포크 마다의 검증 손실값 (val_loss)
  * 매 에포크 마다의 검증 정확도 (val_acc)

* **사용법**

  ```python
  hist = model.fit(X_train, Y_train, epochs=1000, batch_size=10, validation_data=(X_val, Y_val))
  
  print(hist.history['loss'])
  print(hist.history['acc'])
  print(hist.history['val_loss'])
  print(hist.history['val_acc'])
  ```

<br>

### MNIST를 다층 퍼셉트론 신경망 모델로 학습시키는 간단한 예제

```python
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(3)

# 1. 데이터셋 생성하기

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
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
```

**실행 결과**

![image](https://user-images.githubusercontent.com/43431081/80379257-ce14bb00-88d8-11ea-8d92-07438463dba6.png)

> 100번 째 쯤에 손실값이 증가하는 것을 보면 **과적합(overfitting)이** 발생했다고 볼 수 있다.

* **train_loss (노란색)** : 훈련 손실값, x축은 에포크 수, 좌측 y축은 손실값
* **val_loss (빨간색)** : 검증 손실값, x축은 에포크 수, 좌측 y축은 손실값
* **train_acc (파란색)** : 훈련 정확도, x축은 에포크 수, 우측 y축은 정확도
* **val_acc (녹색)** : 검증 정확도, x축은 에포크 수, 우측 y축은 정확도

<br>

## 2. 텐서보드와 연동하기

1. 코드 작성

   ```python
   ...
   
   # 4. 모델 학습시키기
   tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_grads=True,
                                         write_images=True)
   model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data=(x_val, y_val), callbacks=[tb_hist])
   ```

2. 텐서보드 실행

   ```bash
   $ tensorboard --logdir=./graph
   ```

<br>

## 3. 직접 콜백함수 만들어보기

순환신경망 모델인 경우에는 fit() 함수를 여러 번 호출하기 때문에 제대로 학습상태를 볼 수 없다.

<br>

* **순환신경망 모델 코드**

  ```python
  for epoch_idx in range(1000):
    print('epoochs : ' + str(epoch_idx))
    hist = model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=false)
    model.reset_states()
  ```

* **fit 함수를 여러 번 호출하더라도 학습 상태가 유지될 수 있도록 콜백함수 정의**

  ```python
  import keras
  
  # 사용자 정의 히스토리 클래스 정의
  class CustomHistory(Keras.callbacks.Callback):
    def init(self):
      self.train_loss = []
      self.val_loss = []
      self.train_acc = []
      self.val_acc = []
  
    def on_epoch_end(self, batch, logs={}):
      self.train_loss.append(logs.get('loss'))
      self.val_loss.append(logs.get('val_loss'))
      self.train_acc.append(logs.get('acc'))
      self.val_acc.append(logs.get('val_acc'))
  ```

* **콜백 함수를 사용해서 학습하기**

  ```python
  ...
  
  # 4. 모델 학습시키기 (콜백함수 이용)
  custom_hist = CustomHistory()
  custom_hist.init()
  
  for epoch_idx in range(1000):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=10, validation_data=(x_val, y_val),
              callbacks=[custom_hist])
  ```