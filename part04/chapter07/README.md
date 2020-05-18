# 07. 시계열수치입력 수치 예측 모델 레시피

각 모델에 코사인(cosine) 데이터를 학습시킨 후, 처음 일부 데이터를 알려주면 이후 코사인 형태의 데이터를 얼마나 잘 예측하는지 테스트해보자.

<br>

## 데이터셋 준비

먼저 코사인 데이터를 만들어보자.

```python
import numpy as np

signal_data = np.cos(np.arange(1600) * (20 * np.pi / 1000))[:, None]
```

> 시간의 흐름에 따라 진폭이 -1.0 에서 1.0 사이로 변하는 1,600 개의 실수값을 생성한다.

<br>

생성한 데이터를 확인해보자.

```python
%matplotlib inline
import matplotlib.pyplot as plt

plot_x = np.arange(1600)
plot_y = signal_data
plt.plot(plot_x, plot_y)
plt.show()
```

![image](https://user-images.githubusercontent.com/43431081/81267427-6d784180-9081-11ea-8f76-7457a9879f21.png)

<br>

학습을 위해서 데이터와 라벨로 구성된 데이터셋으로 만들어야 한다. 이전 수치들을 입력하여 다음 수치를 예측하는 문제이므로 **데이터는 이전 수치들이 되고, 라벨은 다음 수치가 된다.**

```python
def create_dataset(signal_data, lock_back=1):
  dataX, dataY = [], []
  for i in range(len(signal_data) - lock_back):
    dataX.append(signal_data[i : (i + lock_back), 0])
    dataY.append(signal_data[i + lock_back, 0])
  return np.array(dataX), np.array(dataY)
```

> create_dataset() 함수는 시계열 수치를 입력받아 데이터셋을 생성한다.
>
> lock_back 인자는 얼마만큼의 이전 수치를 데이터를 만들것인가를 결정한다.

<br>

-1.0 에서 1.0 까지의 값을 가지는 코사인 데이터를 0.0과 1.0 사이의 값을 가지도록 정규화를 한 뒤 훈련셋과 시험셋으로 분리한다.

```python
from sklearn.preprocessing import MinMaxScaler

lock_back = 40

# 데이터 전처리
scaler = MinMaxScaler(feature_range(0, 1))
signal_data = scaler.fit_transform(signal_data)

# 데이터 분리
train = signal_data[0:800]
val = signal_data[800:1200]
test = signal_data[1200:]

# 데이터셋 생성
x_train, y_train = create_dataset(trin, lock_back)
x_val, y_val = create_dataset(val, lock_back)
x_test, y_test = create_dataset(test, lock_back)
```

> 이전 40개의 수치를 입력하여 다음 수치 1개를 예측하는 데이터셋을 만들기 위해 lock_back 인자를 40으로 설정한다.
>
> lock_back 인자에 따라 모델의 성능이 달라지므로 적정 값을 지정하는 것이 중요하다.

<br>

## 레이어 준비

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
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_LSTM_s.png" alt="img"></td>
      <td style="text-align: center">LSTM</td>
      <td style="text-align: left">Long-Short Term Memory unit의 약자로 순환 신경망 레이어 중 하나입니다.</td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_tanh_s.png" alt="img"></td>
      <td style="text-align: center">tanh</td>
      <td style="text-align: left">LSTM의 출력 활성화 함수로 사용됩니다.</td>
    </tr>
  </tbody>
</table>

4개의 타임스텝을 가진 LSTM, 출력 활성화 함수로 tanh을 사용한다. 

내부적으로는 모든 블록에서 같은 가중치를 사용하고 있다.

<img src="http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_LSTM_Example_m.png">

<br>

## 모델 준비

### 다층퍼셉트론 모델

Dense 레이어가 4개인 다층퍼셉트론 모델이다. 은닉층에 사용된 Dense 레이어는 32개의 뉴런을 가지고 있고, **relu** 활성화 함수를 사용하였다. 출력층의 Dense 레이어는 하나의 수치값을 예측을 하기 위해서 1개의 뉴런을 가지며, 별도의 활성화 함수를 사용하지 않는다. 과적합을 방지하기 위해 **Dropout 레이어가 삽입된다.**

```python
model = Sequential()
model.add(Dense(32, input_dim=40, activation='relu'))
model.add(Dropout(0.3))
for i in range(2):
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.3))
model.add(Dense(1))
```

<img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_1m.png">

<br>

### 순환신경망 모델

한 개의 LSTM 레이어를 이용하여 순환신경망 모델을 구성하였다. 출력층은 하나의 수치값을 예측하기 위해 1개 뉴런을 가진 Dense 레이어를 사용한다.

```python
model = Sequential()
model.add(LSTM(32, input_shape=(None, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))
```

<img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_2m.png">

<br>

### 상태유지 순환신경망 모델

순환신경망 모델과 동일하나 **'stateful = True'** 옵션을 사용하여 상태유지 가능한 순환신경망 모델을 구성하였다.

상태유지 모드일 경우 한 배치에서 학습된 상태가 다음 배치 학습 시에 전달되는 방식이다.

<img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_3m.png">

<br>

### 상태유지 스택 순환신경망 모델

상태유지 순환신경망을 여러겹 쌓아올린 모델이다. 층이 하나인 순환신경망에 비해 더 깊은 추론이 가능한 모델이다.

```python
model = Sequential()
for i in range(2):
  model.add(LSTM(32, batch_input_shape=(1, lock_back, 1), stateful=True, 
                 return_sequences=True))
  model.add(Dropout(0.3))
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))
```

<img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_4m.png">

<br>

## 전체 소스

### 다층퍼셉트론 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline

def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data) - look_back):
        dataX.append(signal_data[i : (i + look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 40

# 1. 데이터셋 생성하기
signal_data = np.cos(np.arange(1600) * (20 * np.pi / 1000))[:, None]

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(signal_data)

# 데이터 분리
train = signal_data[0:800]
val = signal_data[800:1200]
test = signal_data[1200:]

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_train = np.squeeze(x_train)
x_val = np.squeeze(x_val)
x_test = np.squeeze(x_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(32, input_dim=40, activation='relu'))
model.add(Dropout(0.3))
for i in range(2):
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adagrad')

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, verbose=0)
print('Train Score:', trainScore)
valScore = model.evaluate(x_val, y_val, verbose=0)
print('Validation Score:', valScore)
testScore = model.evaluate(x_test, y_test, verbose=0)
print('Test Score:', testScore)

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0, None]
predictions = np.zeros((look_ahead, 1))
for i in range(look_ahead):
    prediction = model.predict(xhat, batch_size=32)
    predictions[i] = prediction
    xhat = np.hstack([xhat[:, 1:], prediction])

plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label='prediction')
plt.plot(np.arange(look_ahead), y_test[:look_ahead], label='test function')
plt.legend()
plt.show()
```

**실행결과**

```
Train on 760 samples, validate on 360 samples
Epoch 1/200
760/760 [==============================] - 2s 2ms/step - loss: 0.1135 - val_loss: 0.0380
...
Epoch 200/200
760/760 [==============================] - 0s 94us/step - loss: 0.0102 - val_loss: 0.0174
```

![image](https://user-images.githubusercontent.com/43431081/81393180-d8497b80-915a-11ea-90fb-fdd8e7badf0e.png)

![image](https://user-images.githubusercontent.com/43431081/81393192-df708980-915a-11ea-929e-2aa47e3671c2.png)

<br>

### 순환신경망 모델

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline

def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data) - look_back):
        dataX.append(signal_data[i : (i + look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 40

# 1. 데이터셋 생성하기
signal_data = np.cos(np.arange(1600) * (20 * np.pi / 1000))[:, None]

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(signal_data)

# 데이터 분리
train = signal_data[0:800]
val = signal_data[800:1200]
test = signal_data[1200:]

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
model = Sequential()
model.add(LSTM(32, input_shape=(None, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, verbose=0)
model.reset_states()
print('Train Score:', trainScore)
valScore = model.evaluate(x_val, y_val, verbose=0)
model.reset_states()
print('Validataion Score:', valScore)
testScore = model.evaluate(x_test, y_test, verbose=0)
model.reset_states()
print('Test Score:', testScore)

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0]
predictions = np.zeros((look_ahead, 1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:], prediction])

plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label='prediction')
plt.plot(np.arange(look_ahead), y_test[:look_ahead], label='test function')
plt.legend()
plt.show()
```

**실행 결과**

```
Train Score: 0.0001523049422662313
Validataion Score: 0.00015033694087631172
Test Score: 0.00015033694087631172
```

![image](https://user-images.githubusercontent.com/43431081/81396192-e77ef800-915f-11ea-85eb-226850bbddc0.png)

![image](https://user-images.githubusercontent.com/43431081/81396220-f36aba00-915f-11ea-9219-d5a336d2eea5.png)

<br>

### 상태유지 순환신경망 모델

```python
# 0. 사용할 패키지 불러오기
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline

def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data) - look_back):
        dataX.append(signal_data[i : (i + look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
    
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

look_back = 40

# 1. 데이터셋 생성하기
signal_data = np.cos(np.arange(1600) * (20 * np.pi / 1000))[:, None]

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(signal_data)

# 데이터 분리
train = signal_data[0:800]
val = signal_data[800:1200]
test = signal_data[1200:]

# 데이터 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
model = Sequential()
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
custom_hist = CustomHistory()
custom_hist.init()

for i in range(200):
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist],
              validation_data=(x_val, y_val))
    model.reset_states()
    
# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
model.reset_states()
print('Train Score:', trainScore)
valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
model.reset_states()
print('Validation Score:', valScore)
testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
model.reset_states()
print('Test Score:', testScore)

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0]
predictions = np.zeros((look_ahead, 1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:], prediction])

plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label='prediction')
plt.plot(np.arange(look_ahead), y_test[:look_ahead], label='test function')
plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/43431081/81649902-ffaa8c00-946b-11ea-83fb-d2caff8a7ce0.png)

![image](https://user-images.githubusercontent.com/43431081/81649861-f0c3d980-946b-11ea-93ea-5a4f1b998b21.png)

<br>

### 상태유지 스택 순환신경망 모델

```python
# 0. 사용할 패키지 불러오기
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline

def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data) - look_back):
        dataX.append(signal_data[i : (i + look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

look_back = 40

# 1. 데이터셋 생성하기
signal_data = np.cos(np.arange(1600) * (20 * np.pi / 1000))[:, None]

# 데이터 전처리
scaler = MinMaxScaler(feature_range = (0, 1))
signal_data = scaler.fit_transform(signal_data)

# 데이터 분리
train = signal_data[0:800]
val = signal_data[800:1200]
test = signal_data[1200:]

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
model = Sequential()
for i in range(2):
    model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
custom_hist = CustomHistory()
custom_hist.init()

for i in range(200):
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, verbose=2, callbacks=[custom_hist],
              validation_data=(x_val, y_val))
    model.reset_states()
    
# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
model.reset_states()
print('Train Score:', trainScore)
valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
model.reset_states()
print('Validataion Score:', valScore)
testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
model.reset_states()
print('Test Score:', testScore)

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0]
predictions = np.zeros((look_ahead, 1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:], prediction])

plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label='prediction')
plt.plot(np.arange(look_ahead), y_test[:look_ahead], label='test function')
plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/43431081/81639013-b18a8e00-9455-11ea-8d9e-cec380385676.png)

![image](https://user-images.githubusercontent.com/43431081/81639025-b7806f00-9455-11ea-9c75-9c4097e56baf.png)

<br>

## 학습결과 비교

<table>
  <thead>
    <tr>
      <th style="text-align: center">다층퍼셉트론 모델</th>
      <th style="text-align: center">순환신경망 모델</th>
      <th style="text-align: center">상태유지 순환신경망 모델</th>
      <th style="text-align: center">상태유지 스택 순환신경망 모델</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_17_2.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_19_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_21_1.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_23_1.png" alt="img"></td>
    </tr>
    <tr>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_17_4.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_19_3.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_21_3.png" alt="img"></td>
      <td style="text-align: center"><img src="http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_23_3.png" alt="img"></td>
    </tr>
  </tbody>
</table>

