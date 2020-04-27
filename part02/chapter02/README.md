# CHAPTER 02. 학습과정 이야기

케라스에서는 모델을 학습시킬 때 fit() 함수를 사용하는데, 그 인자에 따라 학습과정 및 결과가 차이난다. 학습과정이 어떤 방식으로 일어나는지 살펴보자.

<br>

## 1. 배치사이즈와 에포크

케라스에서 만든 모델을 학습할 때는 fit() 함수를 사용한다.

```python
model.fit(x, y, batch_size=32, epochs=10)
```

* **인자들**
  * **x** : 입력 데이터
  * **y** : 라벨값
  * **batch_size** : 몇 개의 샘플로 가중치를 갱신할 것인지 지정
  * **epochs** : 학습 반복 횟수

<br>

모델은 문제를 푼 뒤 해답과 맞춰봐야 학습이 일어난다. 모델의 결과값과 주어진 라벨값의 오차를 줄이기 위해 **역전파(Backpropagation)** 알고리즘으로 가중치가 갱신된다.

배치사이즈가 작을수록 가중치 갱신이 자주 일어납니다.