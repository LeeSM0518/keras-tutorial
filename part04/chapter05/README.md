# CHAPTER 05. 영상입력 이진분류모델 레시피

숫자 손글씨 데이터셋인 MNIST을 이용하여 홀수/짝수를 구분을 위한 데이터셋을 생성해 보고, 다층퍼셉트론 및 컨볼루션 신경망 모델을 구성하고 학습 시켜보자.

이 모델은 임의의 영상으로부터 A와 B를 구분하는 문제나 양성과 음성을 구분하는 문제를 풀 수 있다.

<br>

## 1. 데이터셋 준비

케라스 함수에서 제공하는 숫자 손글씨 데이터셋인 MNIST을 이용하였다.

초기 라벨값은 0에서 9까지 정수로 지정되어 있다. 데이터 정규화를 위해서 255.0 으로 나누었다.

<br>

다층퍼셉트론 신경망 모델에 입력하기 위해 데이터셋 생성 코드

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, width * height).astype('float32') / 255.0
x_test = x_test.reshape(10000, width * height).astype('float32') / 255.0
```

<br>

컨볼루션 신경망 모델에 입력하기 위해 데이터셋 생성하는 코드

```python
x_train = x_train.reshape(60000, width, height, 1).astype('float32') / 255.0
x_test = x_test.reshape(10000, width, height, 1).astype('float32') / 255.0
```

> 샘플수, 너비, 높이, 채널수로 총 4차원 배열로 구성된다.

<br>

불러온 훈련셋을 다시 훈련셋 50,000개와 검증셋 10,000개로 나눈다.

```python
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]
```

<br>

라벨값은 다중클래스분류로 0에서 9까지 지정되어 있으나 이것을 홀수/짝수로 바꾸어서 이진분류 라벨로 지정해보자.

'1'은 홀수를 의미하고, '0'은 짝수를 의미한다.

```python
y_train = y_train % 2
y_val = y_val % 2
y_test = y_test % 2
```

<br>

만든 데이터셋 일부를 가시화 해보자.

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt_row = 5
plt_col = 5

plt.rcParams['figure.figsize'] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):
  sub_plt = axarr[i//plt_row, i%plt_col]
  sub_plt.axis('off')
  sub_plt.imshow(x_test[i].reshpe(width, height))
  
  sub_plt.title = 'R: '
  
  if y_test[i] :
    sub_plt_title += 'odd '
  else:
    sub_plt_title += 'even '
  
  sub_plt.set_title(sub_plt_title)
  
plt.show()
```

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
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dropout_1D_s.png" alt="img"></td>
      <td style="text-align: center">Dropout</td>
      <td style="text-align: left">과적합을 방지하기 위해서 학습 시에 지정된 비율만큼 임의의 입력 뉴런(1차원)을 제외시킵니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dropout_2D_s.png" alt="img"></td>
      <td style="text-align: center">Dropout</td>
      <td style="text-align: left">과적합을 방지하기 위해서 학습 시에 지정된 비율만큼 임의의 입력 뉴런(2차원)을 제외시킵니다.</td>
    </tr>
  </tbody>
</table>

<br>

## 3. 모델 준비

### 다층퍼셉트론 신경망 모델

```python
model = Sequential()
model.add(Dense(256, input_dim=width * height, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

<img src="http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_0m.png">

<br>

### 컨볼루션 신경망 모델

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

<img src="http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_1m.png">

<br>

### 깊은 컨볼루션 신경망 모델

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

<img src="http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_2m.png">

<br>

## 4. 전체 소스

### 다층퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 1. 데이터셋 생성하기

width = 28
height = 28

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, width * height).astype('float32') / 255.0
x_test = x_test.reshape(10000, width * height).astype('float32') / 255.0

# 훈련셋과 검증셋 분리
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]

# 데이터셋 전처리 : 홀수는 1, 짝수는 0으로 변환
y_train = y_train % 2
y_val = y_val % 2
y_test = y_test % 2

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(256, input_dim=width * height, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 0.5])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)

# 7. 모델 사용하기
yhat_test = model.predict(x_test, batch_size=32)

plt_row = 5
plt_col = 5

plt.rcParams['figure.figsize'] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test[i].reshape(width, height))

    sub_plt_title='R: '

    if y_test[i]:
        sub_plt_title += 'odd '
    else:
        sub_plt_title += 'even '

    sub_plt_title += 'P: '

    if yhat_test[i] >= 0.5:
        sub_plt_title += 'odd '
    else:
        sub_plt_title += 'even '

    sub_plt.set_title(sub_plt_title)

plt.show()
```

**실행 결과**

```
Train on 50000 samples, validate on 10000 samples
Epoch 1/30
50000/50000 [==============================] - 7s 144us/step - loss: 0.3091 - accuracy: 0.8738 - val_loss: 0.1520 - val_accuracy: 0.9467
...
Epoch 30/30
50000/50000 [==============================] - 5s 108us/step - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0717 - val_accuracy: 0.9833

10000/10000 [==============================] - 1s 53us/step
## evaluation loss and metrics ##
[0.05729653679503349, 0.9861999750137329]
```

![image](https://user-images.githubusercontent.com/43431081/81149378-1149ea80-8fb9-11ea-98e9-09661da10f7d.png)

![image](https://user-images.githubusercontent.com/43431081/81149401-1ad35280-8fb9-11ea-908c-f0c324667b9c.png)

<br>

### 컨볼루션 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten

# 1. 데이터셋 생성하기
width = 28
height = 28

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, width, height, 1).astype('float32') / 255.0
x_test = x_test.reshape(10000, width, height, 1).astype('float32') / 255.0

# 훈련셋과 검증셋 분리
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]

# 데이터셋 전처리 : 홀수는 1, 짝수는 0으로 반환
y_train = y_train % 2
y_val = y_val % 2
y_test = y_test % 2	

# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 0.5])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([0.8, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)

# 7. 모델 사용하기
yhat_test = model.predict(x_test, batch_size=32)

%matplotlib inline
import matplotlib.pyplot as plt

plt_row = 5
plt_col = 5

plt.rcParams['figure.figsize'] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):
    sub_plt = axarr[i//plt_row, i%plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test[i].reshape(width, height))

    sub_plt_title = 'R: '

    if y_test[i]:
        sub_plt_title += 'odd '
    else:
        sub_plt_title += 'even '

    sub_plt_title += 'P: '

    if yhat_test[i] >= 0.5:
        sub_plt_title += 'odd '
    else:
        sub_plt_title += 'even '

    sub_plt.set_title(sub_plt_title)

plt.show()
```

**실행 결과**

```
10000/10000 [==============================] - 1s 80us/step
## evaluation loss and metrics ##
[0.023719010082622116, 0.9926999807357788]
```

![image](https://user-images.githubusercontent.com/43431081/81172337-01420300-8fd9-11ea-9df4-d49dd4b76113.png)

<br>

### 깊은 컨볼루션 신경망 모델

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout

# 1. 데이터셋 생성하기

width = 28
height = 28

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, width, height, 1).astype('float32') / 255.0
x_test = x_test.reshape(10000, width, height, 1).astype('float32') / 255.0

# 훈련셋과 검증셋 분리
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]

# 데이터셋 전처리 : 홀수는 1, 짝수는 0
y_train = y_train % 2
y_val = y_val % 2
y_test = y_test % 2

# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 0.5])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([0.8, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6.모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)

# 7. 모델 사용하기
yhat_test = model.predict(x_test, batch_size=32)

%matplotlib inline
import matplotlib.pyplot as plt

plt_row = 5
plt_col = 5

plt.rcParams['figure.figsize'] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):
    sub_plt = axarr[i//plt_row, i%plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test[i].reshape(width, height))

    sub_plt_title = 'R: '
    
    if y_test[i]:
        sub_plt_title += 'odd '
    else:
        sub_plt_title += 'even '
    
    sub_plt_title += 'P: '

    if yhat_test[i] >= 0.5:
        sub_plt_title += 'odd '
    else:
        sub_plt_title += 'even '

    sub_plt.set_title(sub_plt_title)

plt.show()
```

**실행 결과**

```
Train on 50000 samples, validate on 10000 samples
Epoch 1/30
50000/50000 [==============================] - 21s 420us/step - loss: 0.3928 - accuracy: 0.8197 - val_loss: 0.1536 - val_accuracy: 0.9445
...
Epoch 30/30
50000/50000 [==============================] - 15s 305us/step - loss: 0.0147 - accuracy: 0.9948 - val_loss: 0.0318 - val_accuracy: 0.9920

10000/10000 [==============================] - 1s 118us/step
## evaluation loss and metrics ##
[0.019188887589745265, 0.9947999715805054]
```

![image](https://user-images.githubusercontent.com/43431081/81176892-cf349f00-8fe0-11ea-83f8-4ae2d27a0115.png)

![image](https://user-images.githubusercontent.com/43431081/81176938-da87ca80-8fe0-11ea-9700-fe528e942f9f.png)

<br>

## 5. 학습결과 비교

다층퍼셉트론 신경망 모델은 훈련정확도는 검증 손실값은 높아지고 있어 과적합이 발생하였다.

컨볼루션 신경망 모델은 다층퍼셉트론 신경망 모델에 비해 높은 성능을 보이고 있다.

깊은 컨볼루션 신경망 모델은 드롭아웃(Dropout) 레이어 덕분에 과적합이 발생하지 않고 검증 손실값이 지속적으로 떨어지는 것을 확인할 수 있다.

<table>
  <thead>
    <tr>
      <th style="text-align: center">다층퍼셉트론 신경망 모델</th>
      <th style="text-align: center">컨볼루션 신경망 모델</th>
      <th style="text-align: center">깊은 컨볼루션 신경망 모델</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_output_16_2.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_output_18_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_output_20_1.png" alt="img"></td>
    </tr>
  </tbody>
</table>

