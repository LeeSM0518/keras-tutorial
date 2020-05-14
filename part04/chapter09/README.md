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

```

