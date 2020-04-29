# CHAPTER 01. 다층 퍼셉트론 레이어 이야기

케라스의 핵심 데이터 구조는 모델이고 이 모델을 구성하는 것이 레이어이다.

<br>

## 1. 인간의 신경계를 모사한 뉴런 이야기

<img src="http://tykimos.github.io/warehouse/2017-1-27_MLP_Layer_Talk_neuron.png">

신경망에서 사용되는 뉴런은 인간의 신경계를 모사한 것이다.

* **axon(축삭돌기)** : 다른 뉴런의 수상돌기와 연결된다.
* **dendrite(수상돌기)** : 다른 뉴런의 축삭 돌기와 연결된다.
* **synapse(시냅스)** : 축삭돌기와 수상돌기가 연결된 지점이다. 여기서 한 뉴런이 다른 뉴런으로 신호를 전달한다.

<br>

* **뉴런 모델링**
  * *x0, x1, x2* : 입력되는 뉴런의 축삭돌기로부터 전달되는 신호의 양
  * *w0, w1, w2* : 시냅스의 강도, 즉 입력되는 뉴런의 영향력
  * *w0x0 + w1x1 + w2x2* : 입력되는 신호의 양과 해당 신호의 시냅스 강도가 곱해진 값의 합계
  * *f* : 최종 합계가 다른 뉴런에게 전달되는 신호의 양을 결정짓는 규칙, 이를 <u>활성화 함수</u> 라고 부른다.

<br>

## 2. 입출력을 모두 연결해주는 Dense 레이어

* Dense 레이어는 입력과 출력을 모두 연결해준다. 

* 연결선은 **가중치(weight)를** 포함하고 있다. 이 가중치는 연결강도를 의미한다.

* `가중치가 높을수록 해당 입력 뉴런이 출력 뉴런에 미치는 영향이 크고, 낮을수록 미치는 영향이 작다.`

* 입력 뉴런과 출력 뉴런을 모두 연결한다고 해서 전결합층이라고 불리고, 케라스에서는 **Dense라는** 클래스로 구현이 되어 있다.

* **Dense 사용 예제**

  ```python
  Dense(8, input_dim=4, activation='relu')
  ```

  * **첫 번째 인자** : 출력 뉴런의 수를 설정한다.

  * **input_dim** : 입력 뉴런의 수를 설정한다.

  * **activation** : 활성화 함수를 설정한다.

    * **linear** : 디폴트 값, 입력 뉴런과 가중치로 계산된 결과값이 그대로 출력된다.

    * **relu** : rectifier 함수, 은닉층에 주로 쓰인다.

      <img src="https://mlnotebook.github.io/img/transferFunctions/relu.png">

    * **sigmoid** : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰인다.

      <img src="https://mlnotebook.github.io/img/transferFunctions/sigmoid.png">

      

    * **softmax** : 소프트맥스 함수, 다중 클래스분류 문제에서 출력층에 주로 쓰인다.

      <img src="https://t1.daumcdn.net/cfile/tistory/990DA44F5B41DC3705">

      <img src="https://t1.daumcdn.net/cfile/tistory/9919053B5B41E19529">

<br>

Dense 레이어는 입력 뉴런 수에 상관없이 출력 뉴런 수를 자유롭게 설정할 수 있기 때문에 출력층으로 많이 사용한다.

**이진 분류 문제에서** 0과 1을 나타내는 출력 뉴런이 하나만 있으면 되기 때문에 출력 뉴런이 1개이고, 입력 뉴런과 가중치를 계산한 값을 0에서 1사이로 표현할 수 있는 **활성화 함수인 'sigmoid'를** 사용한다.

```python
Dense(1, input_dim=3, activation='sigmoid')
```

<br>

**다중클래스분류 문제에서는** 클래스 수만큼 출력 뉴런이 필요하다. 만약 세 가지 종류로 분류한다면, 출력 뉴런이 3개이고 입력 뉴런과 가중치를 계산한 값을 각 클래스의 확률 개념으로 표현할 수 있는 **활성화 함수인 'softmax'를** 사용한다.

```python
Dense(3, input_dim=4, activation='softmax')
```

<br>

Dense 레이어는 보통 출력층 이전의 **은닉층으로 많이 쓰이고** 영상이 아닌 **수치자료 입력 시에는 입력층으로도 많이 쓰인다.** 이 때 **활성화 함수로 'relu'**가 주로 사용된다. relu는 학습과정에서 **역전파 시에 좋은 성능이** 나오는 것으로 알려져 있다.

```python
Dense(4, input_dim=6, activation='relu')
```

또한 **입력층이 아닐 때에는** 이전층의 출력 뉴런 수를 알 수 있기 때문에 **input_dim을 지정하지 않아도 된다.**

```python
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

<br>

### 실제 케라스 구현

4개의 입력 값을 받아 이진 분류하는 문제를 풀 수 있는 모델

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 케라스 시각화
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```