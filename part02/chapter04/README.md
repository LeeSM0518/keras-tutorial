# CHAPTER 04. 학습 조기종료 시키기

## 조기종료 시키기

학습 조기종료를 위해서는 **'EarlyStopping()'** 이라는 함수를 사용하며, **더 이상의 개선의 여지가 없을 때 학습을 종료시키는 콜백 함수이다.**

먼저 fit() 함수에서 EarlyStopping() 콜백함수를 지정하는 방법은 다음과 같다.

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()
hist = model.fit(x_train, y_train, epochs=3000, batch_size=10, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

<br>

EarlyStopping 콜백함수에서 설정할 수 있는 인자는 다음과 같다.

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                             mode='auto')
```

* **monitor** : 관찰하고자 하는 항목이다.
* **min_delta** : 개선되고 있다고 판단하기 위한 최소 변화랑
* **patience** : 개선이 없다고 바로 종료하지 않고 개선이 없는 에포크를 얼마나 기다려 줄 것인가를 지정
* **verbose** : 얼마나 자세하게 정보를 표시할 것인가를 지정
* **mode** : 관찰 항목에 대해 개선이 없다고 판단하기 위한 기준을 지정한다.
  * auth: 관찰하는 이름에 따라 자동으로 지정한다.
  * min: 관찰하고 있는 항목이 감소되는 것을 멈출 때 종료
  * max: 관찰하고 있는 항목이 증가되는 것을 멈출 때 종료