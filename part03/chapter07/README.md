# CHAPTER 07. 순환 신경망 모델 만들어보기

## 시퀀스 데이터 준비

문자와 숫자로 된 음표로는 모델 입출력으로 사용할 수 없기 때문에 각 코드를 숫자로 변환할 수 있는 사전을 만들어보자.

```python
# c(도), d(레), e(미), f(파), g(솔), a(라), b(시)
# 4(4분음표), 8(8분음표)
code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}
```

* **code2idx** : 코드를 숫자로 변환
* **idx2code** : 숫자를 코드로 변환

<br>

위와 같은 사전을 이용해서 순차적인 음표를 우리가 지정한 윈도우 크기만큼 잘라 데이터셋을 생성하는 함수를 정의해보자.

```python
import numpy as np

def seq2dataset(seq, window_size):
  dataset = []
  for i in range(len(seq) - window_size):
    subset = seq[i:(i + window_size + 1)]
    dataset.append([code2idx[item] for item in subset])
  return np.array(dataset)
```

<br>

seq 라는 변수에 "나비야" 곡 전체 음표를 저장한 다음, seq2dataset() 함수를 호출하여 dataset을 생성한다. 데이터셋은 앞서 정의한 사전에 따라 숫자로 변환되어 생성된다.

```python
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4', 'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4', 'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4', 'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)
print(dataset)
```

```
(50, 5)
[[11  9  2 10  8]
 ...
 [11 11  9  9  2]]
```

<br>

## 학습 과정

첫 4개 음표를 입력하면 나머지를 연주할 수 있는 모델을 만드는 것이 목표다.		

* **학습 시키는 방식**

  <img src="http://tykimos.github.io/warehouse/2017-4-9-RNN_Layer_Talk_5.png">

  * 파란색 박스가 입력값이고, 빨간색 박스가 우리가 원하는 출력값이다.
  * 1~4 번째 음표를 데이터로 5번째 음표를 라벨값으로 학습시킨다.
  * 다음에는 2~5번째 음표를 데이터로 6번째 음표를 라벨값으로 학습시킨다.
  * 이후 한 음표씩 넘어가면서 노래 끝까지 학습시킨다.

<br>

## 예측 과정

`한 스텝 예측` 과 `곡 전체 예측` 이다.

<br>

### 한 스텝 예측

실제 음표 4개를 입력하여 다음 음표를 1개를 예측하는 것을 반복한다.

<img src="http://tykimos.github.io/warehouse/2017-4-9-RNN_Layer_Talk_6.png">

<br>

### 곡 전체 예측

입력된 초가 4개 음표만을 입력으로 곡 전체를 예측하는 것이다.

<img src="http://tykimos.github.io/warehouse/2017-4-9-RNN_Layer_Talk_7.png">

<br>

## 다층 퍼셉트론 신경망 모델

다층 퍼셉트론 신경망 모델을 학습시켜 보자.

Dense 레이어를 3개로 구성하였고, 입력 속성을 4개, 출력을 12개(one_hot_vec_size=12)로 설정했다.

```python
model = Sequential()
model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(one_hot_vec_size, activation='softmax'))
```

"나비야" 악보를 이 모델로 학습하게 되고, 4개의 음표를 입력으로 받고 그 다음 음표가 라벨값으로 지정된다.

<br>

### 전체 소스

```python
# 0. 사용할 패키지 불러오기
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

# 랜덤시드 고정시키기
np.random.seed(5)

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
  def init(self):
    self.losses = []

  def on_epoch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

# 데이터셋 생성 함수
def seq2dataset(seq, window_size):
  dataset = []
  for i in range(len(seq) - window_size):
    subset = seq[i:(i + window_size + 1)]
    dataset.append([code2idx[item] for item in subset])
  return np.array(dataset)

# 1. 데이터 준비하기

# 코드 사전 정의

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}

# 시퀀스 데이터 정의
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 2. 데이터셋 생성하기
dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)
print(dataset)

# 입력(X)과 출력(Y) 변수로 분리하기
x_train = dataset[:, 0:4]
y_train = dataset[:, 4]

max_idx_value = 13

# 입력값 정규화 시키기
x_train = x_train / float(max_idx_value)

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print('one hot encoding vector size is ', one_hot_vec_size)

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = LossHistory()
history.init()

# 5. 모델 학습시키기
model.fit(x_train, y_train, epochs=2000, batch_size=10, verbose=2, callbacks=[history])

# 6. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train)
print('%s : %.2f%%' %(model.metrics_names[1], scores[1] * 100))

# 8. 모델 사용하기
pred_count = 50 # 최대 예측 개수 정의

# 한 스텝 예측
seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(x_train)

for i in range(pred_count):
  idx = np.argmax(pred_out[i])  # one-hot 인코딩을 인덱스 값으로 변환
  seq_out.append(idx2code[idx]) # seq_out 는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장

print('one step prediction : ', seq_out)

# 곡 전체 예측
seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in = [code2idx[it] / float (max_idx_value) for it in seq_in]  # 코드를 인덱스값으로 변환

for i in range(pred_count):
  sample_in = np.array(seq_in)
  sample_in = np.reshape(sample_in, (1, 4)) # batch_size, feature
  pred_out = model.predict(sample_in)
  idx = np.argmax(pred_out)
  seq_out.append(idx2code[idx])
  seq_in.append(idx / float(max_idx_value))
  seq_in.pop(0)

print('full song prediction : ', seq_out)
```

<br>

## 기본 LSTM 모델

이번에는 간단한 기본 LSTM 모델로 테스트 해보자.

* **모델 구성**

  * 128 메모리 셀을 가진 LSTM 레이어 1개와 Dense 레이어로 구성
  * 입력은 샘플 50개, 타임스텝 4개, 속성 1개로 구성
  * 상태유지(stateful) 모드 비활성화

* **모델 구성 방법**

  ```python
  model = Sequential()
  model.add(LSTM(128, input_shape = (4, 1)))
  model.add(Dense(one_hot_vec_size, activation='softmax'))
  ```

<br>

LSTM을 제대로 활용하기 위해서는 상태유지 모드, 배치사이즈, 타임스텝, 속성에 대한 개념에 이해가 필요하다.

* **타임스텝**
  * 하나의 샘플에 포함된 시퀀스 개수
  * input_length, window_size 와 동일
  * 현재 문제에서는 매 샘플마다 4개의 값을 입력하므로 4개로 지정
* **속성**
  * 입력되는 음표 1개당 하나의 인덱스 값을 입력하므로 속성이 1개다.

> LSTM의 인자로 **input_shape = (4, 1)** 이 입력되게 된다.

<br>

LSTM 모델에 따라 입력할 데이터셋도 샘플 수, 타임스텝 수, 속성 수 형식으로 맞추어야 한다. 따라서 **x_train의 형식을 변환해보자.**

```python
x_train = np.reshape(x_train, (50, 4, 1)) # 샘플 수, 타임스텝 수, 속성 수
```

<br>

이 LSTM 모델로 학습할 경우, 다층 퍼셉트론 신경망 모델과 동일하게 4개의 음표를 입력으로 받고, 그 다음 음표가 라벨값으로 지정된다.

다층 퍼셉트론 신경망 모델과 차이점이 있다면, 다층 퍼셉트론 신경망 모델에서는 4개의 음표가 **4개의 속성으로** 입력되고, LSTM에서는 4개의 음표가 4개의 시퀀스 입력으로 들어간다. **여기서 속성은 1개 이다.**

<img src="http://tykimos.github.io/warehouse/2017-4-9-RNN_Layer_Talk_train_LSTM.png">

<br>

### 전체 소스

```python
# 0. 사용할 패키지 불러오기
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

# 랜덤시드 고정시키기
np.random.seed(5)

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 데이터셋 생성 함수
def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq) - window_size):
        subset = seq[i : (i + window_size + 1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)
  
# 1. 데이터 준비하기

# 코드 사전 정의

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}

# 시퀀스 데이터 정의

seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 2. 데이터셋 생성하기

dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)

# 입력(X)과 출력(Y) 변수로 분리하기
x_train = dataset[:, 0:4]
y_train = dataset[:, 4]

max_idx_value = 13

# 입력값 정규화 시키기
x_train = x_train / float(max_idx_value)

# 입력을 (샘플 수, 타임스텝, 특성수)로 형태 변환
x_train = np.reshape(x_train, (50, 4, 1))

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print('one hot encoding vector size is ', one_hot_vec_size)

# 3. 모델 구성하기
model = Sequential()
model.add(LSTM(128, input_shape = (4, 1)))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = LossHistory() # 손실 이력 객체 생성
history.init()

# 5. 모델 학습시키기
model.fit(x_train, y_train, epochs=2000, batch_size=14, verbose=2, callbacks=[history])

# 6. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))

# 8. 모델 사용하기
pred_count = 50 # 최대 예측 개수 저으이

# 한 스텝 예측
seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(x_train)

for i in range(pred_count):
    idx = np.argmax(pred_out[i])   # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx])  # seq_out 는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장

print('one step prediction : ', seq_out)

# 곡 전체 예측
seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in] # 코드를 인덱스값으로 변환

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4, 1)) # 샘플 수, 타임스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)

print('full song prediction : ', seq_out)
```

<br>

## 상태유지 LSTM 모델

이번에는 상태유지(Stateful) LSTM 모델에 대해서 알아보자. 여기서 **상태유지** 라는 것은 <u>현재 학습된 상태가 다음 학습 시 초기 상태로 전달된다는 것을 의미한다.</u>

> 상태유지 모드에서는 현재 샘플의 학습 상태가 다음 샘플의 초기 상태로 전달된다.

<br>

상태유지 모드를 사용하게 되면, 긴 시퀀스 데이터를 샘플 단위로 잘라서 학습하더라도 LSTM 내부적으로 기억할 것은 기억하고, 버릴 것은 버려서 기억해야 할 중요한 정보만 이어갈 수 있도록 상태가 유지된다.

상태유지 LSTM 모델을 생성하기 위해서는 LSTM 레이어 생성 시, **stateful = True** 로 설정한다. 또한 상태유지 모드에서는 입력형태를 **batch_input_shape = (배치사이즈, 타임스텝, 속성)** 으로 설정해야 한다.

```python
model = Sequential()
model.add(LSTM(128, batch_input_shape = (1, 4, 1), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))
```

<br>

상태유지 모드에서는 모델 학습 시, 상태 초기화에 대한 고민이 필요하다. 현재 샘플 학습 상태가 다음 샘플 학습의 초기상태로 상태를 유지시키지 않고 초기화해야 한다. 현재 샘플과 다음 샘플 간의 **순차적인 관계가 없을 경우에는 상태가 유지되지 않고 초기화가 되어야 한다.**

현재 코드에서는 한 곡을 가지고 계속 학습을 시키고 있으므로 새로운 에포크 시작 시에만 상태 초기화를 수행하면 된다.

```python
num_epochs = 2000

for epoch_idx in range(num_epochs):
  print('epochs : ' + str(epoch_idx))
  # 50 is X.shape[0]
  model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, suffle=False)
  model.reset_states()
```

<img src="http://tykimos.github.io/warehouse/2017-4-9-RNN_Layer_Talk_train_stateful_LSTM.png">

<br>

### 전체 소스

```python
# 0. 사용할 패키지 불러오기
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils

# 랜덤시드 고정시키기
np.random.seed(5)

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []
    
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 데이터셋 생성 함수
def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq) - window_size):
        subset = seq[i : (i + window_size + 1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)
  
# 1. 데이터 준비하기

# 코드 사전 정의

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}

# 시퀀스 데이터 정의

seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 2. 데이터셋 생성하기

dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)

# 입력(X)과 출력(Y) 변수로 분리하기

x_train = dataset[:, 0:4]
y_train = dataset[:, 4]

max_idx_value = 13

# 입력값 정규화 시키기
x_train = x_train / float(max_idx_value)

# 입력을 (샘플 수, 타임스텝, 특성 수)로 형태 변환
x_train = np.reshape(x_train, (50, 4, 1))

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print('one hot encoding vector size is ', one_hot_vec_size)

# 3. 모델 구성하기
model = Sequential()
model.add(LSTM(128, batch_input_shape = (1, 4, 1), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습시키기
num_epochs = 2000

history = LossHistory() # 손실 이력 객체 생성

history.init()

for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False,
              callbacks=[history]) # 50 is X.shape[0]
    model.reset_states()
    
# 6. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train, batch_size=1)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))
model.reset_states()

# 8. 모델 사용하기
pred_count = 50 # 최대 예측 개수 정의

# 한 스탭 예측

seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(x_train, batch_size=1)

for i in range(pred_count):
    idx = np.argmax(pred_out[i])   # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx])  # seq_out 는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장 

model.reset_states()

print('one step prediction : ', seq_out)

# 곡 전체 예측

seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in] # 코드를 인덱스값으로 변환

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4, 1)) # 샘플 수, 타입스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)

model.reset_states()

print('full song prediction : ', seq_out)
```

<br>

## 입력 속성이 여러 개인 모델 구성

상태유지 LSTM 모델에서 입력형태를 **batch_input_shape = (배치사이즈, 타임스텝, 속성)** 으로 설정하는데, 마지막 인자를 통해 속성의 개수를 지정할 수 있다.

현재 입력값이 'c4, e4, g8' 등으로 되어 있는데, 이를 음정과 음길이로 나누어서 2개의 속성으로 입력해보자. 즉, **'c4'는 '(c,4)'로** 나누어서 입력하게 되는 것이다.

이를 위해 데이터셋 만드는 함수를 수정해보자.

```python
def code2features(code):
  features = []
  features.append(code2scale[code[0]] / float(max_scale_value))
  features.append(code2length[code[1]])
  return features
```

<br>

LSTM 모델 생성 시 batch_input_shape 인자의 마지막 값이 '1' 에서 '2' 로 수정되었다.

```python
model = Seqeuntial()
model.add(LSTM(128, batch_input_shape = (1, 4 ,2), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))
```

<br>

### 전체 소스

```python
# 0. 사용할 패키지 불러오기
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils

# 랜덤시드 고정시키기
np.random.seed(5)

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 데이터셋 생성 함수
def seq2dataset(seq, window_size):
    dataset_X = []
    dataset_Y = []
    
    for i in range(len(seq) - window_size):
        subset = seq[i : (i + window_size + 1)]
        for si in range(len(subset) - 1):
            features = code2features(subset[si])
            dataset_X.append(features)
        dataset_Y.append([code2idx[subset[window_size]]])
    
    return np.array(dataset_X), np.array(dataset_Y)

# 속성 변환 함수
def code2features(code):
    features = []
    features.append(code2scale[code[0]] / float(max_scale_value))
    features.append(code2length[code[1]])
    return features
  
# 1. 데이터 준비하기

# 코드 사전 정의

code2scale = {'c':0, 'd':1, 'e':2, 'f':3, 'g':4, 'a':5, 'b':6}
code2length = {'4':0, '8':1}

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}

max_scale_value = 6.0
    
# 시퀀스 데이터 정의
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 2. 데이터셋 생성하기

x_train, y_train = seq2dataset(seq, window_size = 4)
print('x_train 1 : ', x_train)
print('y_train 1 : ', y_train)

# 입력을 (샘플 수, 타임스텝, 특성 수)로 형태 변환
x_train = np.reshape(x_train, (50 ,4, 2))
print('x_train 2 : ', x_train)

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)
print('y_train 2 : ', y_train)

one_hot_vec_size = y_train.shape[1]

print('one hot encoding vector size is ', one_hot_vec_size)

# 3. 모델 구성하기
model = Sequential()
model.add(LSTM(128, batch_input_shape = (1, 4, 2), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습시키기
num_epochs = 2000

history = LossHistory() # 손실 이력 객체 생성
history.init()

for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False,
              callbacks=[history]) # 50 is X.shape[0]
    model.reset_states()
    
# 6. 학습과정 살펴보기
%matplotlib inline 
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train, batch_size=1)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))
model.reset_states()

# 8. 모델 사용하기
pred_count = 50 # 최대 예측 개수 정의

# 한 스텝 예측

seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(x_train, batch_size=1)

for i in range(pred_count):
    idx = np.argmax(pred_out[i])  # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx]) # seq_out 는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장

print('one step prediction : ', seq_out)

model.reset_states()

# 곡 전체 예측
seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in

seq_in_features = []

for si in seq_in:
    features = code2features(si)
    seq_in_features.append(features)

for i in range(pred_count):
    sample_in = np.array(seq_in_features)
    sample_in = np.reshape(sample_in, (1, 4, 2)) # 샘플 수, 타임스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    
    features = code2features(idx2code[idx])
    seq_in_features.append(features)
    seq_in_features.pop(0)

model.reset_states()

print('full song prediction : ', seq_out)
```