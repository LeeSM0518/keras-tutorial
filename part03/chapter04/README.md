# CHAPTER 04. 컨볼루션 신경망 모델 만들어보기

## 1. 문제 정의하기

컨볼루션 신경망 모델에 적합한 문제는 이미지 기반의 분류이다.

* **문제 형태** : 다중 클래스 분류
* **입력** : 손으로 그린 삼각형, 사각형, 원 이미지
* **출력** : 삼각형, 사각형, 원일 확률을 나타내는 벡터

<br>

**우선 필요한 패키지를 불러오고, 랜덤 시드를 명시**

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 랜덤시드 고정시키기
np.random.seed(3)
```

<br>

## 2. 데이터 준비하기

http://tykimos.github.io/warehouse/2017-3-8_CNN_Getting_Started_handwriting_shape.zip

<br>

## 3. 데이터셋 생성하기

* **ImageDataGenerator 클래스**
  * 이미지 파일을 쉽게 학습시킬 수 있도록 케라스가 제공하는 클래스
  * 데이터 부풀리기(data augmentation)을 위해 막강한 기능을 제공한다.

<br>

먼저 **ImageDataGenerator 클래스를** 이용하여 객체를 생성한 뒤 **flow_from_directory() 함수를** 호출하여 **제너레이터(generator)를 생성한다.**

<br>

* **flow_from_directory() 함수의 주요 인자**
  * **첫 번째 인자** : 이미지 경로를 지정한다.
  * **target_size** : 패치 이미지 크기를 지정한다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절된다.
  * **batch_size** : 배치 크기를 지정한다.
  * **class_mode** : 분류 방식에 대하여 지정한다.
    * **categorical** : 2D one-hot 부호화된 라벨이 반환된다.
    * **binary** : 1D 이진 라벨이 반환된다.
    * **sparse** : 1D 정수 라벨이 반환된다.
    * **None** : 라벨이 반환되지 않는다.

<br>

```python
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  '파일 경로',
  target_size=(24, 24),
  batch_size = 3,
  class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
  '파일 경로',
  target_size=(24, 24),
  batch_size=3,
  class_mode='categorical')
```

<br>

## 4. 모델 구성하기

컨볼루션 신경망 모델을 구성해보자.

* **각 레이어들**

  1. **컨볼루션 레이어** : 입력 이미지 크기 24x24, 입력 이미지 채널 3개, 필터 크기 3x3, 필터 수 32개, 활성화 함수 'relu'
  2. **컨볼루션 레이어** : 필터 크기 3x3, 필터 수 64개, 활성화 함수 'relu'
  3. **맥스풀링 레이어** : 풀 크기 2x2
  4. **플래튼 레이어**
  5. **댄스 레이어** : 출력 뉴런 수 128개, 활성화 함수 'relu'
  6. **댄스 레이어** : 출력 뉴런 수 3개, 활성화 함수 'softmax'

* **코드**

  ```python
  model=Sequential()
  model.add(Conv2D(32, kerner__size(3, 3),
                   activation='relu'),
            input_shape=(24, 24, 3))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(3, activation='softmax'))
  ```

<br>

## 5. 모델 학습과정 설정하기

모델을 정의하였다면 모델을 손실함수와 최적화 알고리즘으로 엮는다.

* **loss** : 현재 가중치 세트를 평가하는데 사용한 손실함수이다. 다중 클래스 문제이므로 'categorical_crossentropy'으로 지정한다.
* **optimizer** : 최적의 가중치를 검색하는데 사용하는 최적화 알고리즘으로 효율적인 경사 하강법 알고리즘 중 하나인 'adam'을 사용한다.
* **metrics** : 평가 척도를 나타내며 분류 문제에서는 일반적으로 'accuracy'으로 지정한다.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

<br>

## 6. 모델 학습시키기

케라스에서는 모델을 학습시킬 때 주로 fit() 함수를 사용하지만 제너레이터로 생성된 배치로 학습시킬 경우에는 **fit_generator()** 함수를 사용한다.

```python
model.fit_generator(
  train_generator,
  steps_per_epoch=15,
  epochs=50,
  validation_data=test_generator,
  validation_steps=5)
```

* **첫 번째 인자** : 훈련 데이터셋을 제공할 제너레이터를 지정
* **steps_per_epoch** : 한 epoch에 사용할 스텝 수를 지정
* **epochs** : 전체 훈련 데이터셋에 대해 학습 반복 횟수를 지정
* **validation_data** : 검증 데이터셋을 제공할 제너레이터를 지정
* **validation_steps** : 한 epoch 종료 시마다 검증할 때 사용되는 검증 스탭 수를 지정

<br>

## 7. 모델 평가하기

제너레이터에서 제공되는 샘플로 평가할 때는 **evaluate_generator** 함수를 사용한다.

```python
scores = model.evaluate_generator(test_generator, steps=5)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))
```

<br>

## 8. 모델 사용하기

모델 사용 시에 제너레이터에서 제공되는 샘플을 입력할 때는 **predict_generator** 함수를 사용한다. 예측 결과는 클래스별 확률 벡터로 출력된다. 제너레이터의 'class_indices'를 출력하면 해당 열의 클래스명을 알 수 있다.

```python
output = model.predict_generator(test_generator, step=5)
np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}' .format(x)})
```

<br>

## 9. 전체 소스

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 랜덤시드 고정시키기
np.random.seed(3)

# 1. 데이터 생성하기
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/My Drive/Colab Notebooks/handwriting_shape/train',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/content/drive/My Drive/Colab Notebooks/handwriting_shape/test',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical')

# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24, 24, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit_generator(
    train_generator,
    steps_per_epoch=15,
    epochs=50,
    validation_data=test_generator,
    validation_steps=5)

# 6. 모델 평가하기
print('-- Evaluate --')
scores = model.evaluate_generator(test_generator, steps=5)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))

# 7. 모델 사용하기
print('-- Predict --')
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}' .format(x)})
print(test_generator.class_indices)
print(output)
```

**실행 결과**

```
# 6
-- Evaluate --
accuracy: 100.00%

# 7
-- Predict --
{'circle': 0, 'rectangle': 1, 'triangle': 2}
[[0.000 0.000 1.000]
 [0.996 0.000 0.004]
 [0.000 0.000 1.000]
 [1.000 0.000 0.000]
 [0.000 1.000 0.000]
 [1.000 0.000 0.000]
 [0.000 1.000 0.000]
 [0.000 1.000 0.000]
 [0.000 0.003 0.997]
 [0.069 0.798 0.133]
 [0.000 0.001 0.999]
 [1.000 0.000 0.000]
 [1.000 0.000 0.000]
 [0.000 0.000 1.000]
 [0.000 1.000 0.000]]
```