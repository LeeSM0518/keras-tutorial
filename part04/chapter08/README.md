# CHAPTER 08. 문장(시계열수치)입력 이진분류 모델 레시피

모델을 입력하기 위해서 문장을 시계열수치로 인코딩하는 방법과 여러 가지 이진분류모델을 구성해 보고, 학습 결과를 살펴보자.

이 모델은 문장 혹은 시계열수치로 양성/음성을 분류하거나 이벤트 발생 유무를 감지하는 문제를 풀 수 있다.

<br>

## 1. 데이터셋 준비

IMDB에서 제공하는 영화 리뷰 데이터셋을 이용해보자. 이 데이터셋은 훈련셋 25,000개, 시험셋 25,000개의 샘플을 제공한다.

라벨은 1과 0으로 좋아요/싫어요로 지정된다.

```python
from keras.datasets import imdb
# num_words: 20,000번째로 많이 사용하는 단어까지만 데이터셋을 만들기 위해
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
```

<br>

훈련셋의 데이터가 어떻게 구성되어 있는지 살펴보자.

```python
print(x_train[0])
```

```
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 19193, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 10311, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 12118, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
```

* 리뷰 문자을 시계열수치로 바꾼 데이터이다.

<br>

훈련셋 25,000개를 다시 훈련셋 20,000개와 검증셋 5,000개로 분리하자.

```python
x_val = x_train[20000:]
y_val = y_train[20000:]
x_train = x_train[:20000]
y_train = y_train[:20000]
```

<br>

모델의 입력으로 사용하려면 고정된 길이로 만들어야 하므로 케라스에서 제공되는 전처리 함수인 **sequence의 pad_sequences() 함수를** 사용한다.

```python
from keras.preprocessing import sequence

x_train = sequence.pad_sequences(x_train, maxlen=200)
x_val = sequence.pad_sequences(x_val, maxlen=200)
x_test = sequence.pad_sequences(x_test, maxlen=200)
```

* **pad_sequences() 함수 역할**
  * 문장의 길이를 maxlen 인자로 맞춰준다.
  * (num_samples, num_timesteps)으로 2차원의 numpy 배열로 만들어준다.
    * maxlen을 200으로 지정하면 num_timesteps도 200이 된다.

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
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Embedding_s.png" alt="img"></td>
      <td style="text-align: center">Embedding</td>
      <td style="text-align: left">단어를 의미론적 기하공간에 매핑할 수 있도록 벡터화시킵니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Conv1D_s.png" alt="img"></td>
      <td style="text-align: center">Conv1D</td>
      <td style="text-align: left">필터를 이용하여 지역적인 특징을 추출합니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_GlobalMaxPooling1D_s.png" alt="img"></td>
      <td style="text-align: center">GlobalMaxPooling1D</td>
      <td style="text-align: left">여러 개의 벡터 정보 중 가장 큰 벡터를 골라서 반환합니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_MaxPooling1D_s.png" alt="img"></td>
      <td style="text-align: center">MaxPooling1D</td>
      <td style="text-align: left">입력벡터에서 특정 구간마다 값을 골라 벡터를 구성한 후 반환합니다.</td>
    </tr>
  </tbody>
</table>

<br>

## 3. 모델 준비

### 다층퍼셉트론 신경망 모델

임베딩 레이어로 인코딩한 후 Dense 레이어를 통해 분류하는 다층퍼셉트론 신경망 모델

<img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_1m.png">

```python
model = Sequential()
model.add(Embedding(20000, 128, input_length=200))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

* **임베딩 레이어의 인자**
  * 첫 번째 인자(input_dim): 단어 사전의 크기를 말하며 총 20,000 개의 단어 종류가 있다는 의미이다.
  * 두 번째 인자(output_dim): 단어를 인코딩한 후 나오는 벡터 크기이다. 이 값이 128이라면 단어를 128차원의 의미론적 기하공간에 나타낸다는 의미이다.
  * input_length: 단어의 수 즉 문자의 길이를 나타낸다.

<br>

### 순환 신경망 모델

문장을 단어들의 시퀀스로 간주하고 순환(LSTM) 레이어의 입력으로 구성한 모델이다.

<img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_2m.png">

```python
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
```

* 임베딩 레이어 다음에 LSTM 레이어가 오는 경우에는 임베딩 레이어에 input_length 인자를 따로 설정할 필요는 없다.
  * 입력 문장의 길이에 따라 input_length가 자동으로 정해지고, 이것이 LSTM 레이어에는 timesteps으로 입력되기 때문이다.

<br>

### 컨볼루션 신경망 모델

문장 해석에 컨볼루션(Conv1D) 레이어를 이용한 모델이다.

글로벌 맥스플링(GlobalMaxPoolin1D) 레이어는 컨볼루션 레이어가 문장을 훑어가면서 나온 특징벡터들 중 가장 큰 벡터를 골라준다.

즉, 문맥을 보면서 주요 특징을 뽑아내고, 그 중 가장 두드러지는 특징을 고르는 것이다.

<img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_3m.png">

```python
model = Sequential()
model.add(Embedding(20000, 128, input_length=200))
model.add(Dropout(0.2))
model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
```

<br>

### 순환 컨볼루션 신경망 모델

컨볼루션 레이어에서 나온 특징벡터들을 맥스풀링(MaxPooling1D)를 통해 1/4로 줄여준 다음 LSTM의 입력으로 넣어주는 모델이다.

<img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_4m.png">

```python
model = Sequential()
model.add(Embedding(20000, 128, input_length=200))
model.add(Dropout(0.2))
model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
```

<br>

## 4. 전체 소스

### 다층퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Flatten

max_features = 20000
text_max_words = 200

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words=max_features)

# 훈련셋과 검증셋 분리
x_val = x_train[20000:]
y_val = y_train[20000:]
x_train = x_train[:20000]
y_train = y_train[:20000]

# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([-0.2, 1.2])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([-0.2, 1.2])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)
```

![image](https://user-images.githubusercontent.com/43431081/81897057-cb140d00-95f0-11ea-812e-546983153b32.png)

```
25000/25000 [==============================] - 8s 314us/step
## evaluation loss and metrics ##
[0.42208416783332825, 0.8512399792671204]
```

<br>

### 순환 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Flatten

max_features = 20000
text_max_words = 200

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 훈련셋과 검증셋 분리
x_val = x_train[20000:]
y_val = y_train[20000:]
x_train = x_train[:20000]
y_train = y_train[:20000]

# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([-0.2, 1.2])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([-0.2, 1.2])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)
```

![image](https://user-images.githubusercontent.com/43431081/81898241-41197380-95f3-11ea-9cb3-dbb76d739275.png)

```
25000/25000 [==============================] - 55s 2ms/step
## evaluation loss and metrics ##
[0.33199588078498843, 0.866320013999939]
```

<br>

### 컨볼루션 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Flatten, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D

max_features = 20000
text_max_words = 200

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 훈련셋과 검증셋 분리
x_val = x_train[20000:]
y_val = y_train[20000:]
x_train = x_train[:20000]
y_train = y_train[:20000]

# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Dropout(0.2))
model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([-0.2, 1.2])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([-0.2, 1.2])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)
```

![image](https://user-images.githubusercontent.com/43431081/81900068-1c26ff80-95f7-11ea-953d-b8593f17ce48.png)

```
25000/25000 [==============================] - 19s 763us/step
## evaluation loss and metrics ##
[0.2783369765043259, 0.8837199807167053]
```

<br>

### 순환 컨볼루션 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, \
                         Flatten, Dropout, \
                         Conv1D, MaxPooling1D

max_features = 20000
text_max_words = 200

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 훈련셋과 검증셋 분리
x_val = x_train[20000:]
y_val = y_train[20000:]
x_train = x_train[:20000]
y_train = y_train[:20000]

# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Dropout(0.2))
model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([-0.2, 1.2])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([-0.2, 1.2])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)
```

![image](https://user-images.githubusercontent.com/43431081/81901684-e5061d80-95f9-11ea-865b-f9871baecf22.png)

```
25000/25000 [==============================] - 36s 1ms/step
## evaluation loss and metrics ##
[0.35505525747299194, 0.8609200119972229]
```

<br>

## 5. 학습결과 비교

<table>
  <thead>
    <tr>
      <th style="text-align: center">다층퍼셉트론 신경망 모델</th>
      <th style="text-align: center">순환 신경망 모델</th>
      <th style="text-align: center">컨볼루션 신경망 모델</th>
      <th style="text-align: center">순환 컨볼루션 신경망 모델</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_output_15_2.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_output_17_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_output_19_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_output_21_1.png" alt="img"></td>
    </tr>
  </tbody>
</table>