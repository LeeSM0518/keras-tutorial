# CHAPTER 05. 평가 이야기

문제에 따라 단순히 정확도로만 평가하기 힘든 경우가 있다. 조금 더 알아보면 **민감도, 특이도, 재현율 등의** 용어가 나오게 된다.

<br>

## 1. 분류하기

* **정확도**
  * 전체 중에 양성을 양성이라 고르고 음성을 음성이라고 고른 개수의 비율
  * 양성에 대해서만 잘 골라낼 수 있는 능력을 평가하기에는 정확도가 아닌 **민감도를** 봐야 한다.
* **민감도**
  * 양성을 양성이라고 판정을 잘 할수록 민감도가 높다.
  * `민감도 = 판정한 것 중 실제 양성 수 / 전체 양성 수`
  * 음성에 대해서만 잘 골라낼 수 있는 능력을 평가하기에는 민감도가 아닌 **특이도를** 봐야 한다.
* **특이도**
  * 음성을 음성이라고 판정을 잘 할수록 특이도가 높다.
  * `특이도 = 판정한 것 중 실제 음성 수 / 전체 음성 수`

<br>

### 좀 더 살펴보기

확률로 판정결과를 나타내기 위해서 50%가 기준이 되어, 50% 이상이면 홀수 블록이다라고 얘기한다. 여기서 이 50%를 **임계값(threshold)** 라고 부른다.

**ROC(Receiver Operating Characteristic) curve** 는 민감도와 특이도가 어떤 관계를 가지고 변하는지를 그래프로 표시한 것이다.

이러한 ROC curve 아래 면적을 구한 값을 **AUC(Area Under Curve)** 라고 하는데, 하나의 수치로 계산되어서 성능 비교를 간단히 할 수 있다.

ROC curve를 그리는 방법은 간단하다. 각 임계값별로 민감도와 특이도를 계산하여 x축을 특이도, y축을 민감도로 둬서 이차원 평면 상에 점을 찍고 연결하면 된다.

* **ROC curve를 그리는 코드**

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  %matplotlib inline
  
  sens_F = np.array([1.0,  1.0, 1.0,  1.0, 0.75,  0.5,  0.5, 0.5, 0.5, 0.5, 0.0])
  spec_F = np.array([0.0, 0.16, 0.5, 0.66, 0.66, 0.66, 0.83, 1.0, 1.0, 1.0, 1.0])
  
  sens_G = np.array([1.0,  1.0, 0.75, 0.75, 0.5,  0.5,  0.5,  0.5, 0.25, 0.25, 0.0])
  spec_G = np.array([0.0, 0.33, 0.33,  0.5, 0.5, 0.66, 0.66, 0.83, 0.83,  1.0, 1.0])
  
  plt.title('Receiver Operating Characteristic')
  plt.xlabel('False Positive Rate(1 - Specificity)')
  plt.ylabel('True Positive Rate(Sensitivity)')
  
  plt.plot(1-spec_F, sens_F, 'b', label = 'Model F')   
  plt.plot(1-spec_G, sens_G, 'g', label = 'Model G') 
  plt.plot([0,1],[1,1],'y--')
  plt.plot([0,1],[0,1],'r--')
  
  plt.legend(loc='lower right')
  plt.show()
  ```

  **실행 결과**

  ![image](https://user-images.githubusercontent.com/43431081/80438002-14e8cc00-893e-11ea-9026-5e8fe66c5a57.png)

  * 노란점선: 이상적인 모델. 임계값과 상관없이 민감도와 특이도가 100%. AUC 값은 1이다.
  * 빨간점선: 기준선으로서 AUC 값이 0.5 이다.
  * 모델 F와 모델 G를 비교해보면 모델 F가 모델 G보다 상위에 있고 AUC가 면적이 더 넓은 것을 확인할 수 있다.
  * **sklearn 패키지는** ROC curve 및 AUC를 좀 더 쉽게 구할 수 있는 함수를 제공한다.

<br>

* **sklearn 패키지를 이용한 소스코드**

  ```python
  import matplotlib.pyplot as plt
  from sklearn.metrics import roc_curve, auc
  
  class_F = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
  proba_F = np.array([0.05, 0.15, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.95, 0.95])
  
  class_G = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
  proba_G = np.array([0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.65, 0.75, 0.85, 0.95])
  
  false_positive_rate_F, true_positive_rate_F, thresholds_F = roc_curve(class_F, proba_F)
  false_positive_rate_G, true_positive_rate_G, thresholds_G = roc_curve(class_G, proba_G)
  roc_auc_F = auc(false_positive_rate_F, true_positive_rate_F)
  roc_auc_G = auc(false_positive_rate_G, true_positive_rate_G)
  
  plt.title('Receiver Operating Characteristic')
  plt.xlabel('False Positive Rate(1 - Specificity)')
  plt.ylabel('True Positive Rate(Sensitivity)')
  
  
  plt.plot(false_positive_rate_F, true_positive_rate_F, 'b', label='Model F (AUC = %0.2f)'% roc_auc_F)
  plt.plot(false_positive_rate_G, true_positive_rate_G, 'g', label='Model G (AUC = %0.2f)'% roc_auc_G)
  plt.plot([0,1],[1,1],'y--')
  plt.plot([0,1],[0,1],'r--')
  
  plt.legend(loc='lower right')
  plt.show()
  ```

  **실행 결과**

  ![image](https://user-images.githubusercontent.com/43431081/80438322-e15a7180-893e-11ea-829f-54664208e044.png)

<br>

## 2. 검출 및 검색하기

* **정밀도**
  * 모델이 얼마나 정밀한가?
  * 진짜 양성만을 잘 고를수록 정밀도가 높다.
  * `정밀도 = 실제 양성 수 / 양성이라고 판정한 수`
* **재현율**
  * 양성인 것을 놓치지 않고 골라내는가?
  * 양성을 많이 고를수록 재현율이 높다.
  * `재현율 = 검출 양성 수 / 전체 양성 수`

<br>

검출 문제에서 더 좋은 모델을 선별하기 위한 것이 **Precision-Recall Graph** 이다. 이 그래프는 x축을 재현율로 y축을 정밀도로 두어 이차원 평면 상에 결과를 표시한다.

* **그래프를 그리기 위한 소스코드**

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  %matplotlib inline
  
  precision_F = np.array([0.33, 0.38, 0.45, 0.55, 0.57, 0.40, 0.66, 1.0, 1.0, 1.0, 1.0])
  recall_F = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.4, 0.4, 0.4, 0.4, 0.2, 0.0])
  
  precision_G = np.array([0.33, 0.38, 0.36, 0.37, 0.33, 0.40, 0.33, 0.5, 1.0, 1.0, 1.0])
  recall_G = np.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.0])
  
  plt.title('Precision-Recall Graph')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  
  plt.plot(recall_F, precision_F, 'b', label = 'Model F')   
  plt.plot(recall_G, precision_G, 'g', label = 'Model G') 
  
  plt.legend(loc='upper right')
  plt.show()
  ```

  **실행 결과**

  ![image](https://user-images.githubusercontent.com/43431081/80440012-c427a200-8942-11ea-8d7f-fc96cc5a91c3.png)

  * 이러한 그래프를 하나의 수치로 나타낸 것이 **AP(Average Precision)** 이라고 한다. 이는 각 재현율에 해당하는 정밀도를 합한 다음 평균을 취한 것이다.
  * sklearn 패키지는 Precision-Recall Graph 및 AP를 좀 더 쉽게 구할 수 있는 함수를 제공한다.

* **sklearn 패키지를 이용한 소스코드**

  ```python
  import matplotlib.pyplot as plt
  from sklearn.metrics import precision_recall_curve, average_precision_score
  
  class_F = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1])
  proba_F = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.35, 0.35, 0.45, 0.45, 0.55, 0.55, 0.65, 0.85, 0.95])
  
  class_G = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
  proba_G = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.25, 0.35, 0.35, 0.45, 0.55, 0.55, 0.65, 0.75, 0.95])
  
  precision_F, recall_F, _ = precision_recall_curve(class_F, proba_F)
  precision_G, recall_G, _ = precision_recall_curve(class_G, proba_G)
  
  ap_F = average_precision_score(class_F, proba_F)
  ap_G = average_precision_score(class_G, proba_G)
  
  plt.title('Precision-Recall Graph')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  
  plt.plot(recall_F, precision_F, 'b', label = 'Model F (AP = %0.2F)'%ap_F)   
  plt.plot(recall_G, precision_G, 'g', label = 'Model G (AP = %0.2F)'%ap_G)  
  
  plt.legend(loc='upper right')
  plt.show()
  ```

  **실행 결과**

  ![image](https://user-images.githubusercontent.com/43431081/80440218-4021ea00-8943-11ea-895e-1635245ae5da.png)

  * F 모델이 G모델보다 AP 수치가 높으므로 더 좋은 모델이라고 볼 수 있다.

<br>

## 3. 분할하기

* 어떤 모델의 결과가 가장 정확한지 판단하는 방법은 **픽셀 정확도(Pixel Accuracy)** 를 가지고 판단할 수 있다.
  * `Pixel Accuracy = (녹색 블록 맞춘 수 + 노란색 블록 맞춘 수) / 전체 블록 수`

* 색상별로 어느 정도 맞춰야 좋은 평가를 얻게 하려면 어떻게 해야할까? 클래스 별로 픽셀 정확도를 계산하는 방법인 **평균 정확도(Mean Accuracy)** 를 사용하면 된다.
  * 이는 색상별로 픽셀 정확도를 계산하여 평균을 구한 것이다.
  * `Mean Accuracy = (녹색 블록 맞춘 수 / 전체 녹색 블록 수 + 노란색 블록 맞춘 수 / 전체 노란색 블록 수) / 2`
* 틀린 블록에 대한 패널티는 어떻게 고려할까? 이를 고려한 평가방법으로 **MeanIU** 라는 것이 있다.
  * IU는 **Intersection over Union** 의 약자로 특정 색상에서의 실제 블록과 예측 블록 간의 합집합 영역 대비 교집합 영역의 비율이다.
  * MeanIU는 색상별로 구한 IU의 평균값을 취한 것이다.
  * `MeanIU = (녹색 블록 IU + 노란색 블록 IU) / 2`
* 만약 클래스별로 픽셀 수가 다를 경우, 픽셀 수가 많은 클래스에 더 비중을 주고 싶다면 **Frequency Weighted IU** 를 사용한다.