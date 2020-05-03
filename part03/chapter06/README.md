# CHAPTER 06. 순환 신경망 레이어 이야기

순환 신경망 모델은 순차적인 자료에서 규칙적인 패턴을 인식하거나 그 의미를 추론할 수 있다.

케라스에서 제공하는 순환 신경망 레이어는 SimpleRNN, GRU, LSTM 이 있으나 주로 **LSTM을** 사용한다.

<br>

## 1. 긴 시퀀스를 기억할 수 있는 LSTM(Logn Short-Term Memory units) 레이어

* **LSTM 레이어**

  * **입력 형태**

    ```python
    LSTM(3, input_dim=1)
    LSTM(3, input_dim=1, input_length=4)
    ```

    * **첫 번째 인자** : 메모리 셀의 개수
    * **input_dim** : 입력 속성 수
    * **input_length** : 시퀀스 데이터의 입력 길이
    
  * **출력 형태**
  
    * **return_sequences** : 시퀀스 출력 여부
    * LSTM 레이어를 여러 개로 쌓아 올릴 때는 <u>return_sequence=True</u> 옵션을 사용한다.
  
  * **상태유지(stateful) 모드**
  
    * **stateful** : 상태 유지 여부