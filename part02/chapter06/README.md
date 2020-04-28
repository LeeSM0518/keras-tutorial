# CHAPTER 06. 학습 모델 보기 / 저장하기 / 불러오기

우리는 모델 구성과 가중치만 저장해 놓으면, 필요할 때 저장한 모델 구성과 가중치를 불러와서 사용하면 된다.

<br>

## 1. 간단한 모델 살펴보기

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터셋 전처리
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 훈련셋과 검증셋 분리
x_val = x_train[:42000]
x_train = x_train[42000:]
y_val = y_train[:42000]
y_train = y_train[42000:]

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

# 5. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 사용하기
xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]
yhat = model.predict_classes(xhat)

for i in range(5):
  print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
```

**실행 결과**

```
# 4. 모델 학습
Train on 18000 samples, validate on 42000 samples
Epoch 1/5
18000/18000 [==============================] - 2s 121us/step - loss: 1.1547 - accuracy: 0.7248 - val_loss: 0.6578 - val_accuracy: 0.8403
Epoch 2/5
18000/18000 [==============================] - 2s 105us/step - loss: 0.5202 - accuracy: 0.8674 - val_loss: 0.4782 - val_accuracy: 0.8729
Epoch 3/5
18000/18000 [==============================] - 2s 106us/step - loss: 0.4165 - accuracy: 0.8867 - val_loss: 0.4179 - val_accuracy: 0.8853
Epoch 4/5
18000/18000 [==============================] - 2s 104us/step - loss: 0.3709 - accuracy: 0.8988 - val_loss: 0.3828 - val_accuracy: 0.8932
Epoch 5/5
18000/18000 [==============================] - 2s 105us/step - loss: 0.3433 - accuracy: 0.9049 - val_loss: 0.3651 - val_accuracy: 0.8966
<keras.callbacks.callbacks.History at 0x7f79a10dc9e8>

# 5. 모델 평가
10000/10000 [==============================] - 0s 22us/step

loss_and_metrics : [0.33839519055485723, 0.9057999849319458]

# 6. 모델 사용
True : 6, Predict : 0
True : 4, Predict : 4
True : 2, Predict : 2
True : 1, Predict : 1
True : 5, Predict : 5
```

<br>

## 2. 실무에서의 딥러닝 시스템

* **딥러닝 시스템 구성**

  <img src="http://tykimos.github.io/warehouse/2017-6-10-Model_Load_Save_1.png">

<br>

## 3. 학습된 모델 저장하기

모델은 크게 모델 아키텍처와 모델 가중치로 구성된다.

* **모델 아키텍처** : 모델이 어떤 층으로 어떻게 쌓여있는지에 대한 모델 구성
* **모델 가중치** : 처음에는 임의의 값으로 초기화되어 있지만 훈련셋으로 학습하면서 갱신된다.

학습된 모델을 저장한다는 것은 **'모델 아키텍처' 와 '모델 가중치'를 저장한다는 말이다.**

케라스에서는 **save() 함수로** 모델 아키텍처와 모델 가중치를 **'h5' 파일 형식으로 모두 저장할 수 있다.**

<br>

* **전체 소스코드**

  ```python
  # 0. 사용할 패키지 불러오기
  from keras.utils import np_utils
  from keras.datasets import mnist
  from keras.models import Sequential
  from keras.layers import Dense, Activation
  import numpy as np
  from numpy import argmax
  
  # 1. 데이터셋 생성하기
  
  # 훈련셋과 시험셋 불러오기
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  # 데이터셋 전처리
  x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
  x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
  
  # 원핫인코딩 (one-hot encoding) 처리
  y_train = np_utils.to_categorical(y_train)
  y_test = np_utils.to_categorical(y_test)
  
  # 훈련셋과 검증셋 분리
  x_val = x_train[:42000]   # 훈련셋의 30%를 검증셋으로 사용
  x_train = x_train[42000:]
  y_val = y_train[:42000]   # 훈련셋의 30%를 검증셋으로 사용
  y_train = y_train[42000:] 
  
  # 2. 모델 구성하기
  model = Sequential()
  model.add(Dense(units=64, input_dim=28*28, activation='relu'))
  model.add(Dense(units=10, activation='softmax'))
  
  # 3. 모델 학습과정 설정하기
  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  
  # 4. 모델 학습시키기
  model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))
  
  # 5. 모델 평가하기
  loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
  print('')
  print('loss_and_metrics : ' + str(loss_and_metrics))
  
  # 6. 모델 저장하기
  from keras.models import load_model
  model.save('mnist_mlp_model.h5')
  ```

<br>

## 4. 모델 아키텍처 보기

**model_to_dat() 함수를** 통해 모델 아키텍처를 가시화할 수 있다. model 객체를 생성한 뒤라면 언제든지 모델 아키텍처를 블록 형태로 볼 수 있다.

```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

%matplotlib inline
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```

<br>

## 5. 학습된 모델 불러오기

`mnist_mlp_model.h5' 에는 모델 아키텍처와 학습된 모델 가중치가 저장되어 있으니, 이를 불러와 사용해보자.

<br>

1. 모델을 불러오는 함수를 이용하여 저장한 모델 파일로부터 모델을 재형성한다.
2. 실제 데이터로 모델을 사용한다. **predict_classes()** 함수를 통해 가장 확률이 높은 클래스의 인덱스를 알려준다.

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 실무에 사용할 데이터 준비하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_test = np_utils.to_categorical(y_test)
xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]

# 2. 모델 불러오기
from keras.models import load_model
model = load_model('mnist_mlp_model.h5')

# 3. 모델 사용하기
yhat = model.predict_classes(xhat)

for i in range(5):
  print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
```

**실행 결과**

```
True : 9, Predict : 4
True : 7, Predict : 7
True : 7, Predict : 2
True : 8, Predict : 8
True : 0, Predict : 0
```