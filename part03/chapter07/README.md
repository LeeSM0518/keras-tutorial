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

## 5. 다층 퍼셉트론 신경망 모델

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

## 6. 기본 LSTM 모델

이번에는 간단한 기본 LSTM 모델로 테스트 해보자.

* **모델 구성**

  * 128 메모리 셀을 가진 LSTM 레이어 1개와 Dense 레이어로 구성
  * 입력은 샘플 50개, 타임스탭 4개, 속성 1개로 구성
  * 상태유지(stateful) 모드 비활성화

* **모델 구성 방법**

  ```python
  model = Sequential()
  model.add(LSTM(128, input_shape = (4, 1)))
  model.add(Dense(one_hot_vec_size, activation='softmax'))
  ```

<br>

LSTM을 