# CHAPTER 02. 다층 퍼셉트론 신경망 모델 만들어보기

케라스를 이용하여 간단한 다층 퍼셉트론 신경망 모델 만들기

<br>

## 1. 문제 정의하기

이진 분류 예제에 적합한 데이터셋은 8개 변수와 당뇨병 발병 유무가 기록된 '피마족 인디언 당뇨병 발병 데이터셋'이 있다.

<br>

데이터셋을 준비하기에 앞서, 매번 실행 시마다 결과가 달라지지 않도록 **랜덤 시드를 명시적으로 지정한다.**

이것을 하지 않으면 매번 실행 시마다 동일 모델임에도 불구하고 다른 결과가 나오기 때문에 **연구 개발 단계에서 파라미터 조정이나 데이터셋에 따른 결과 차이를 보려면 랜덤 시드를 지정해줘야 한다.**

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 랜덤시드 고정시키기
np.random.seed(5)
```

<br>

## 2. 데이터 준비하기

* **pima-indians-diabetes.names**
  * **인스턴스 수** : 768개
  * **속성 수** : 8가지
  * **클래스 수** : 2가지

<br>

numpy 패키지에서 제공하는 loadtxt() 함수를 통해 데이터를 불러온다.

```python
dataset = np.loadtxt('파일경로', delimiter=',')
```

<br>

## 3. 데이터셋 생성하기

CSV 형식의 파일은 numpy 패키지에서 제공하는 loadtxt() 함수로 직접 불러올 수 있다. 데이터셋에는 속성값과 판정결과가 모두 포함되어 있기 때문에 변수로 분리한다.

```python
x_train = dataset[:700, 0:8]
y_train = dataset[:700, 8]
x_test = dataset[700:, 0:8]
y_test = dataset[700:, 8]
```

<br>

## 4. 모델 구성하기

Dense 레이어만을 사용하여 다층 퍼셉트론 신경망 모델을 구성할 수 있다.

1. *Dense 레이어는 은닉층(hidden layer)으로 8개 뉴런을 입력받아 12개 뉴런을 출력한다.*
2. *Dense 레이어는 은닉층으로 12개 뉴런을 입력받아 8개 뉴런을 출력한다.*
3. *Dense 레이어는 출력 레이어로 8개 뉴런을 입력받아 1개 뉴런을 출력한다.*

```python
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

* 출력 레이어의 활성화 함수를 **'sigmoid'** 로 사용했기 때문에 , 0과 1사이의 값이 출력된다.

<br>

## 5. 모델 학습과정 설정하기

모델을 손실함수와 최적화 알고리즘으로 엮어보자.

* **loss**
  * 현재 가중치 세트를 평가하는데 사용한 손실 함수이다.
  * 이 예제는 이진 클래스 문제이므로 <u>'binary_crossentropy'</u> 으로 지정한다.
* **optimizer**
  * 최적의 가중치를 검색하는데 사용하는 최적화 알고리즘으로 효율적인 경사 하강법 및 알고리즘 중 하나인 <u>'adam'</u> 을 사용한다.
* **metrics**
  * 평가 척도를 나타내며 분류 문제에서는 일반적으로 **'accuracy'** 으로 지정한다.

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

<br>

## 6. 모델 학습시키기

모델을 학습시키기 위해서 **fit()** 함수를 사용한다.

* **첫 번째 인자**
  * 입력 변수 (X)
* **두 번째 인자**
  * 출력 변수 (Y)
* **epochs**
  * 전체 훈련 데이터셋에 대해 학습 반복 횟수
* **batch_size**
  * 가중치를 업데이트할 배치 크기를 의미.

```python
model.fit(x_train, y_train, epochs=15, batch_size=64)
```

<br>

## 7. 모델 평가하기

시험셋으로 학습한 모델을 평가한다.

```python
scores = model.evaluate(x_test, y_test)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))
```

<br>

## 8. 전체 소스

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 랜덤시드 고정시키기
np.random.seed(5)

# 1. 데이터 준비하기
dataset = np.loadtxt('/content/drive/My Drive/Colab Notebooks/pima-indians-diabetes.csv', delimiter=',')

# 2. 데이터셋 생성하기
x_train = dataset[:700, 0:8]
y_train = dataset[:700, 8]
x_test = dataset[700:, 0:8]
y_test = dataset[700:, 8]

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 4. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습시키기
model.fit(x_train, y_train, epochs=1500, batch_size=64)

# 6. 모델 평가하기
scores = model.evaluate(x_test, y_test)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))
```