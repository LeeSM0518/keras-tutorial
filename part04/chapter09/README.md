# CHAPTER 09. 문장(시계열수치)입력 다중클래스분류 모델 레시피

## 1. 데이터셋 준비

뉴스와이어(뉴스 보도 자료) 데이터셋을 이용한다. 이 데이터셋은 총 11,228개의 샘플로 구성되어 있고, 라벨은 46개 주제로 지정되어 0에서 45의 값을 가지고 있다.

데이터셋은 이미 정수로 인코딩되어 있으며, 정수값은 단어의 빈도수를 나타낸다.

```python
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=15000)
```

> 훈련셋과 시험셋의 비율은 load_data() 함수의 test_split 인자로 조절 가능하다. 각 샘플은 뉴스 한 건을 의미하며, 단어의 인덱스로 구성된다.

<br>

훈련셋 8,892개 중 다시 7,000개를 훈련셋으로, 나머지를 검증셋으로 분리한다.

```python
x_val = x_train[7000:]
y_val = y_train[7000:]
x_train = x_train[:7000]
y_train = y_train[:7000]
```

<br>

각 샘플의 길이가 달라서 모델의 입력으로 사용하기 위해 케라스에서 제공되는 전처리 함수인 sequence의 pad_sequences() 함수를 사용한다.

* **pad_sequences() 함수 역할**
  * 문장의 길이를 maxlen 인자로 맞춘다.
  * (num_samples, num_timesteps) 으로 2차원의 numpy 배열로 만들어준다.

```python
from keras.preprocessing import sequence

x_train = sequence.pad_sequences(x_train, maxlen=120)
x_val = sequence.pad_sequences(x_val, maxlen=120)
x_test = sequence.pad_sequences(x_test, maxlen=120)
```

<br>

## 2. 레이어 준비

이전에 "문장입력 이진분류모델"에서 출력층의 활성화 함수만 다르므로 새롭게 소개되는 블록은 없다.

<br>

## 3. 모델 준비

### 다층퍼셉트론 신경망 모델

임베딩 레이어는 0에서 45의 정수값으로 지정된 단어를 128벡터로 인코딩한다. 문장의 길이가 120이므로 임베딩 레이어는 18 속성을 가진 벡터 120개를 반환한다. 이를 플래튼 레이어를 통해 1차원 벡터로 만든 뒤 전결합층으로 전달한다. 

46개 주제를 분류해야 하므로 출력층의 활성화 함수로 **'softmax'를** 사용했다.

```python
model = Sequential()
model.add(Embedding(15000, 128, input_length=120))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(46, activation='softmax'))
```

![image](https://user-images.githubusercontent.com/43431081/81909489-a4140600-9605-11ea-9945-5bd2deedfabc.png)

<br>

### 순환 신경망 모델

임베딩 레이어에서 반환되는 120개 벡터를 LSTM의 타임스텝으로 입력하는 모델이다. LSTM의 input_dim은 임베딩 레이어에서 인코딩된 벡터 크기인 128이다.

```python
model = Sequential()
model.add(Embedding(15000, 128))
model.add(LSTM(128))
model.add(Dense(46, activation='softmax'))
```

![image](https://user-images.githubusercontent.com/43431081/81909505-aa09e700-9605-11ea-8737-6ee806b3c706.png)

<br>

### 컨볼루션 신경망 모델

임베딩 레이어에서 반환되는 120개 벡터를 컨볼루션 필터를 적용한 모델이다.

```python
model = Sequential()
model.add(Embedding(15000, 128, input_length=120))
model.add(Dropout(0.2))
model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPoolin1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(46, activation='softmax'))
```

![image](https://user-images.githubusercontent.com/43431081/81909803-1258c880-9606-11ea-8d13-d650ca0b1350.png)

> 필터크기가 3인 컨볼루션 레이어는 120개의 벡터를 입력받아 118개의 벡터를 반환한다.
>
> 벡터 크기는 컨볼루션 레이어를 통과하면서 128개에서 256개로 늘어났다
>
> 글로벌 맥스풀링 레이어는 입력되는 118개 벡터 중 가장 큰 벡터 하나를 반환한다.
>
> 그 벡터 하나를 전결합층을 통하여 다중 클래스로 분류한다.

<br>

### 순환 컨볼루션 신경망 모델

컨볼루션 레이어에서 나온 특징벡터들을 맥스풀링을 통해 1/4로 줄여준 다음 LSTM의 입력으로 넣어주는 모델이다.

```python
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Dropout(0.2))
model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(46, activation='softmax'))
```

![image](https://user-images.githubusercontent.com/43431081/81909813-15ec4f80-9606-11ea-8164-16ba97039b0b.png)

<br>

## 4. 전체 소스

### 다층퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.datasets import reuters
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

max_features = 15000
text_max_words = 120

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)

# 훈련셋과 검증셋 분리
x_val = x_train[7000:]
y_val = y_train[7000:]
x_train = x_train[:7000]
y_train = y_train[:7000]

# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# one-hot 인코딩
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(46, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([0.0, 1.0])

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

![image](https://user-images.githubusercontent.com/43431081/81930971-4c38c780-9624-11ea-8163-b07ef343cfd4.png)

```
2246/2246 [==============================] - 0s 196us/step
## evaluation loss and metrics ##
[1.4612263896480162, 0.687889575958252]
```

<br>

### 순환 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.datasets import reuters
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

max_features = 15000
text_max_words = 120

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)

# 훈련셋과 검증셋 분리
x_val = x_train[7000:]
y_val = y_train[7000:]
x_train = x_train[:7000]
y_train = y_train[:7000]

# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# one-hot 인코딩
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128))
model.add(Dense(46, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

## 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([0.0, 1.0])

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

![image](https://user-images.githubusercontent.com/43431081/81933287-172e7400-9628-11ea-863b-7f79eb60603f.png)

```
2246/2246 [==============================] - 3s 1ms/step
## evaluation loss and metrics ##
[1.6154488617038472, 0.6665182709693909]
```

<br>

### 컨볼루션 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.datasets import reuters
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Flatten, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D

max_features = 15000
text_max_words = 120

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)

# 훈련셋과 검증셋 분리
x_val = x_train[7000:]
y_val = y_train[7000:]
x_train = x_train[:7000]
y_train = y_train[:7000]

# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# one-hot 인코딩
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Dropout(0.2))
model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(46, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([0.0, 1.0])

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

![image](https://user-images.githubusercontent.com/43431081/81933379-3cbb7d80-9628-11ea-8bfd-43bc365bb812.png)

```
2246/2246 [==============================] - 1s 455us/step
## evaluation loss and metrics ##
[1.2994123129364856, 0.7617987394332886]
```

<br>

### 순환 컨볼루션 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.datasets import reuters
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, \
                         Flatten, Dropout, \
                         Conv1D, MaxPooling1D

max_features = 15000
text_max_words = 120

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)

# 훈련셋과 검증셋 분리
x_val = x_train[7000:]
y_val = y_train[7000:]
x_train = x_train[:7000]
y_train = y_train[:7000]

# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# one-hot 인코딩
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Dropout(0.2))
model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(46, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([0.0, 1.0])

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

![image](https://user-images.githubusercontent.com/43431081/81934626-3b8b5000-962a-11ea-866f-d10a5d3fb338.png)

```
2246/2246 [==============================] - 2s 904us/step
## evaluation loss and metrics ##
[1.4812380506433978, 0.6874443292617798]
```

<br>

## 5. 학습결과 비교

단순한 다층퍼셉트론 신경망 모델보다는 순환 레이어나 컨볼루션 레이어를 이용한 모델의 성능이 더 높아졌다.

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
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Multiclass_Classification_Model_Recipe_output_12_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Multiclass_Classification_Model_Recipe_output_14_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Multiclass_Classification_Model_Recipe_output_16_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Multiclass_Classification_Model_Recipe_output_18_1.png" alt="img"></td>
    </tr>
  </tbody>
</table>



