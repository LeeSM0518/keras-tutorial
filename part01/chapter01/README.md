# CHAPTER 01. 케라스 이야기

케라스(Keras)는 파이썬으로 구현된 쉽고 간결한 딥러닝 라이브러리이다.

직관적인 API로 쉽게 다층퍼셉트론 신경망 모델, 컨볼루션 신경망 모델, 순환 신경망 모델 또는 이를 조합한 모델은 물론 다중 입력 또는 다중 출력 등 다양한 구성을 할 수 있다.

<br>

## 2. 케라스 주요 특징

* **모듈화 (Modularity)**
  * 케라스에서 제공하는 모듈은 독립적이고 설정 가능하며, 최소한의 제약사항으로 연결될 수 있다.
* **최소주의 (Minimalism)**
  * 각 모듈은 짧고 간결하다.
* **쉬운 확장성**
  * 새로운 클래스나 함수로 모듈을 아주 쉽게 추가할 수 있다.
* **파이썬 기반**
  * 파이썬 코드로 모델을이 정의된다.

<br>

## 3. 케라스 기본 개념

케라스의 가장 핵심적인 데이터 구조는 바로 **모델이다.**

<br>

케라스로 딥러닝 모델 만드는 순서

1. **데이터셋 생성하기**
   * 원본 데이터를 불러오거나 시뮬레이션을 통해 데이터 생성
   * 데이터로부터 훈련셋, 검증셋, 시험셋 생성
   * 딥러닝 모델의 학습 및 평가를 할 수 있도록 포맷 변환
2. **모델 구성하기**
   * 시퀀스 모델을 생성한 뒤 필요한 레이어를 추가하여 구성
   * 좀 더 복잡한 모델이 필요할 때는 케라스 함수 API 사용
3. **모델 학습과정 설정하기**
   * 학습에 대한 설정
   * 손실 함수 및 최적화 방법 정의
   * 케라스에서 <u>compile()</u> 함수 사용
4. **모델 학습시키기**
   * 구성한 모델을 훈련셋으로 학습시킨다.
   * 케라스에서 <u>fit()</u> 함수 사용
5. **학습과정 살펴보기**
   * 모델 학습 시 훈련셋, 검증셋의 손실 및 정확도 측정
   * 반복 횟수에 따른 손실 및 정확도 추이를 보면서 학습 상황 판단
6. **모델 평가하기**
   * 준비된 시험셋으로 학습한 모델을 평가
   * 케라스에서 <u>evaluate()</u> 함수를 사용
7. **모델 사용하기**
   * 임의의 입력으로 모델의 출력
   * 케라스에서 <u>predict()</u> 함수를 사용

<br>

### 손글씨 영상을 분류하는 모델 구현

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 1. 데이터셋 생성하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=5, batch_size=32)

# 5. 학습과정 살펴보기
print('## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['acc'])

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# 7. 모델 사용하기
xhat = x_test[0:1]
yhat = model.predict(xhat)
print('## yhat ##')
print(yhat)
```

**실행 결과**

```
# 4
Epoch 1/5
60000/60000 [==============================] - 6s 93us/step - loss: 0.6703 - accuracy: 0.8300
Epoch 2/5
60000/60000 [==============================] - 4s 65us/step - loss: 0.3452 - accuracy: 0.9030
Epoch 3/5
60000/60000 [==============================] - 4s 65us/step - loss: 0.2992 - accuracy: 0.9159
Epoch 4/5
60000/60000 [==============================] - 4s 65us/step - loss: 0.2718 - accuracy: 0.9238
Epoch 5/5
60000/60000 [==============================] - 4s 65us/step - loss: 0.2511 - accuracy: 0.9295

# 5
## training loss and acc ##
[0.670283979678154, 0.3452173143903414, 0.29920490324497223, 0.2717914918879668, 0.25107039091388383]
[0.83003336, 0.9030333, 0.91588336, 0.9238167, 0.92948335]

# 6
10000/10000 [==============================] - 1s 51us/step
## evaluation loss and metrics ##
[0.23610463756024838, 0.9337999820709229]

# 7
## yhat ##
[[3.2247286e-04 1.2165529e-07 1.1787481e-03 3.8814684e-03 1.2738786e-06
  1.4799365e-04 4.4413312e-08 9.9315429e-01 1.9899799e-04 1.1146235e-03]]
```

