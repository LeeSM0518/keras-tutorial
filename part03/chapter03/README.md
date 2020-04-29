# CHAPTER 03. 컨볼루션 신경망 레이어 이야기

컨볼루션 신경망은 이미지가 가지고 있는 특성이 고려되어 설계된 신경망이기에 영상 처리에 주로 사용된다.

* **컨볼루션 신경망 모델의 주요 레이어**
  * 컨볼루션(Convolution) 레이어
  * 맥스풀링(Max Pooling) 레이어
  * 플래튼(Flatten) 레이어

<br>

## 1. 필터로 특징을 뽑아주는 컨볼루션(Convolution) 레이어

영상 처리에 주로 사용되는 Conv2D 레이어를 살펴보자.

* **Conv2D 클래스 사용 예제**

  ```python
  Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')
  ```

  * **첫 번째 인자** : 컨볼루션 필터의 수
  * **두 번째 인자** : 컨볼루션 커널의 (행, 열) 이다.
  * **padding** : 경계 처리 방법
    * **'valid'** : 유효한 영역만 출력한다. 따라서 출력 이미지 사이즈는 입력 이미지 사이즈보다 작다.
    * **'same'** : 출력이미지 사이즈가 입력 이미지 사이즈와 동일하다.
  * **input_shape** : 샘플 수를 제외한 입력 형태
    * **(행, 열 채널 수)** 로 정의한다(흑백: 1, 컬러: 3).
  * **activation** : 활성화 함수를 설정한다.
    * **'linear'** : 디폴트 값, 
    * **'relu'** : rectifier 함수. 은닉층에 주로 쓰인다.
    * **'sigmoid'** : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰인다.
    * **'softmax'** : 소프트맥스 함수, 다중 클래스분류 문제에서 출력층에 주로 사용

  * **입력 형태**

    * image_data_format이 'channels_first'인 경우 (샘플 수, 채널 수, 행, 열)로 이루어진 4D 텐서이다.

    * image_data_format이 'channels_last'인 경우 (샘플 수, 행, 열, 채널 수)로 이루어진 4D 텐서이다.

      > image_data_format 온셥은 "keras.json" 파일 안에 있는 설정이다.

  * **출력 형태**

    * image_data_format이 'channels_first'인 경우 (샘플 수, 필터 수, 행, 열)로 이루어진 4D 텐서이다.
    * image_data_format이 'channels_last'인 경우 (샘플 수, 행, 열, 필터 수)로 이루어진 4D 텐서이다.
    * 행과 열의 크기는 padding이 'same'인 경우에는 입력 형태의 행과 열의 크기가 동일하다.

<br>

* **가중치의 수**

  * 영상도 결국에는 픽셀의 집합이므로 입력 뉴런이 9개 (3x3)이고, 출력 뉴런이 4개 (2x2)인 Dense 레이어로 표현할 수 있다.

    ```python
    Dense(4, input_dim=9)
    ```

    <img src="http://tykimos.github.io/warehouse/2017-1-27_CNN_Layer_Talk_lego_2.png">

  * 컨볼루션 레이어에서의 뉴런 상세 구조

    <img src="http://tykimos.github.io/warehouse/2017-1-27_CNN_Layer_Talk_lego_3.png">

    * Dense 레이어와 비교했을 때, 가중치가 많이 줄어든 것을 볼 수 있다.

<br>

* **경계 처리 방법**
  * 깊은 층을 가진 모델인 경우 'valid'일 때 특징맵이 계속 작아져서 정보가 많이 손실되므로 필터를 통과하더라도 원본 사이즈가 유지될 수 있도록 <u>'same'</u> 으로 설정한다.

<br>

* **필터 수**

  ```python
  Conv2D(3, (2, 2), padding='same', input_shape(3, 3, 1))
  ```

  <img src="http://tykimos.github.io/warehouse/2017-1-27_CNN_Layer_Talk_lego_6.png">

  * 입력 이미지 사이즈가 3 x 3 이다.
  * 2 x 2 커널을 가진 필터가 3개 이다. 가중치는 총 12개 이다.
  * 출력 이미지 사이즈가 3 x 3 이고 총 3개이다. 이는 채널이 3개라고도 표현한다.

<br>

## 2. 사소한 변화를 무시해주는 맥스풀링(Max Pooling) 레이어

컨볼루션 레이어의 출력 이미지에서 주요값만 뽑아 크기가 작은 출력 영상을 만듭니다. 이것은 지역적인 사소한 변화가 영향을 미치지 않도록 한다.

```python
MaxPooling2D(pool_size=(2, 2))
```

* **pool_size** : 수직, 수평 축소 비율을 지정한다.

<br>

## 3. 영상을 일차원으로 바꿔주는 플래튼(Flatten) 레이어

컨볼루션 레이어나 맥스풀링 레이어는 주로 2차원 자료를 다루지만 **전결합층에 전달하기 위해서는 1차원 자료로** 바꿔줘야 한다. 이 때 사용되는 것이 **플래튼 레이어** 이다.

```python
Flatten()
```

* 이전 레이어의 출력 정보를 이용하여 입력 정보는 자동으로 설정된다.
* 출력 형태는 입력 형태에 따라 자동으로 계산되기 때문에 별도로 사용자가 파라미터를 지정해주 읺아도 된다.