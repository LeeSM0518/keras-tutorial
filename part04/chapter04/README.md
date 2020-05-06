# CHAPTER 04. 영상입력 수치 예측 모델 레시피

영상을 입력해서 수치를 예측하는 모델들에 대하여 알아보자.

실습 할 모델은 고정된 지역에서 촬영된 영상으로부터 복잡도, 밀도 등을 수치화하는 문제를 풀 수 있다.

<br>

## 1. 데이터셋 준지

너비가 16, 높이가 16이고, 픽셀값으로 0과 1을 가지는 영상을 만들어보자. 여기서 임의의 값을 라벨값으로 지정했다.

```python
width = 16
height = 16

def generate_dataset(samples):
  ds_x = []
  ds_y = []
  
  for it in range(samples):
    num_pt = np.random.randint(0, width * height)
    img = generate_image(num_pt)
    ds_y.append(num_pt)
    ds_x.append(img)
    
  return np.array(ds_x), np.array(ds_y).reshape(samples, 1)

def generate_image(points):
  img = np.zeros((width, height))
  pts = np.random.random((points, 2))
  
  for ipt in pts:
    img[int(ipt[0] * width), int(ipt[1] * height)] = 1
    
  return img.reshape(width, height, 1)
```

<br>

데이터셋으로 훈련셋을 1500개, 검증셋을 300개, 시험셋을 100개 생성한다.

```python
x_train, y_train = generate_dataset(1500)
x_val, y_val = generate_dataset(300)
x_test, y_test = generate_dataset(100)
```

<br>

만든 데이터셋 일부를 가시화해 보자.

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt_row = 5
plt_col = 5

plt.rcParams['figure.figsize'] = (10, 10)
f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_train[i].reshape(width, height))
    sub_plt.set_title('R ' + str(y_train[i][0]))

plt.show()
```

![image](https://user-images.githubusercontent.com/43431081/81141501-cd021e80-8fa7-11ea-9d77-b2684d844625.png)

> R(Real)은 픽셀값이 1인 픽셀 수 를 의미한다.

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
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dataset2D_s.png" alt="img"></td>
      <td style="text-align: center">2D Input data</td>
      <td style="text-align: left">2차원의 입력 데이터입니다. 주로 영상 데이터를 의미하며, 너비, 높이, 채널수로 구성됩니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Conv2D_s.png" alt="img"></td>
      <td style="text-align: center">Conv2D</td>
      <td style="text-align: left">필터를 이용하여 영상 특징을 추출하는 컨볼루션 레이어입니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_MaxPooling2D_s.png" alt="img"></td>
      <td style="text-align: center">MaxPooling2D</td>
      <td style="text-align: left">영상에서 사소한 변화가 특징 추출에 크게 영향을 미치지 않도록 해주는 맥스풀링 레이어입니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Flatten_s.png" alt="img"></td>
      <td style="text-align: center">Flatten</td>
      <td style="text-align: left">2차원의 특징맵을 전결합층으로 전달하기 위해서 1차원 형식으로 바꿔줍니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_relu_2D_s.png" alt="img"></td>
      <td style="text-align: center">relu</td>
      <td style="text-align: left">활성화 함수로 주로 Conv2D 은닉층에 사용됩니다.</td>
    </tr>
  </tbody>
</table>

<br>

## 3. 모델 준비

### 다층퍼셉트론 신경망 모델

```python
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=width * height))
model.add(Dense(256, activation='relu'))
model.add(Dense(256))
model.add(Dense(1))
```

<img src="http://tykimos.github.io/warehouse/2017-8-20-Image_Input_Numerical_Prediction_Model_Recipe_1m.png">

<br>

### 컨볼루션 신경망 모델

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
```

<img src="http://tykimos.github.io/warehouse/2017-8-20-Image_Input_Numerical_Prediction_Model_Recipe_2m.png">

<br>

## 4. 전체 소스

### 다층퍼셉트론 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

width = 16
height = 16

def generate_dataset(samples):
    ds_x = []
    ds_y = []
    for it in range(samples):
        num_pt = np.random.randint(0, width * height)
        img = generate_image(num_pt)

        ds_y.append(num_pt)
        ds_x.append(img)
    return np.array(ds_x), np.array(ds_y).reshape(samples, 1)

def generate_image(points):
    img = np.zeros((width, height))
    pts = np.random.random((points, 2))
    for ipt in pts:
        img[int(ipt[0] * width), int(ipt[1] * height)] = 1
    return img.reshape(width, height, 1)
  
# 1. 데이터셋 생성하기
x_train, y_train = generate_dataset(1500)
x_val, y_val = generate_dataset(300)
x_test, y_test = generate_dataset(100)

x_train_1d = x_train.reshape(x_train.shape[0], width * height)
x_val_1d = x_val.reshape(x_val.shape[0], width * height)
x_test_1d = x_test.reshape(x_test.shape[0], width * height)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=width * height))
model.add(Dense(256, activation='relu'))
model.add(Dense(256))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mse', optimizer='adam')

# 4. 모델 학습시키기
hist = model.fit(x_train_1d, y_train, batch_size=32, epochs=1000, validation_data=(x_val_1d, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 300.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
score = model.evaluate(x_test_1d, y_test, batch_size=32)

print(score)

# 7. 모델 사용하기
yhat_test = model.predict(x_test_1d, batch_size=32)

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
    sub_plt.set_title('R %d P %.1f' %(y_test[i][0], yhat_test[i][0]))

plt.show()
```

**실행 결과**

```
100/100 [==============================] - 0s 115us/step
90.07034088134766
```

![image](https://user-images.githubusercontent.com/43431081/81144241-9e3b7680-8fae-11ea-93e9-469cd6ac0119.png)

![image](https://user-images.githubusercontent.com/43431081/81144259-a5fb1b00-8fae-11ea-9be0-3448472c4f28.png)

<br>

다층퍼셉트론 신경망 모델의 입력층인 Dense 레이어는 일차원 벡터로 데이터를 입력 받기 때문에, **이차원인 영상을 일차원 벡터로 변환하는 과정이** 필요하다.

```python
x_train_1d = x_train.reshape(x_train.shape[0], width * height)
x_val_1d = x_val.reshape(x_val.shape[0], width * height)
x_test_1d = x_test.reshape(x_test.shape[0], width * height)
```

<br>

### 컨볼루션 신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

width = 16
height = 16

def generate_dataset(samples):
    ds_x = []
    ds_y = []
    for it in range(samples):
        num_pt = np.random.randint(0, width * height)
        img = generate_image(num_pt)
        ds_y.append(num_pt)
        ds_x.append(img)
    return np.array(ds_x), np.array(ds_y).reshape(samples, 1)

def generate_image(points):
    img = np.zeros((width, height))
    pts = np.random.random((points, 2))
    for ipt in pts:
        img[int(ipt[0] * width), int(ipt[1] * height)] = 1
    return img.reshape(width, height, 1)

# 1. 데이터셋 생성하기
x_train, y_train = generate_dataset(1500)
x_val, y_val = generate_dataset(300)
x_test, y_test = generate_dataset(100)

# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mse', optimizer='adam')

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 300.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)

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
    sub_plt.set_title('R %d P %.1f' %(y_test[i][0], yhat_test[i][0]))
```

<br>

## 5. 학습결과 비교

다층퍼셉트론 신경망 모델과 컨볼루션 신경망 모델을 비교했을 때, 현재 파라미터로는 다층퍼셉트론 신경망 모델의 정확도가 더 높았다. 즉, 컨볼루션 신경망 모델이 크게 성능을 발휘하지 못했다.