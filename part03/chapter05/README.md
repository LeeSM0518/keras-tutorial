# CHAPTER 05. 컨볼루션 신경망 모델을 위한 데이터 부풀리기

훈련셋이 부족하거나 훈련셋이 시험셋의 특성을 충분히 반영하지 못할 때, 케라스에서 제공하는 함수를 사용하면 모델의 성능을 크게 향상 시킬수 있다.

<br>

## 데이터 부풀리기

케라스에서는 ImageDataGenerator 함수를 통하여 데이터 부풀리기 기능을 제공한다.

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                            samplewise_center=False,
                                            featurewist_std_normalization=False,
                                            samplewise_std_normalization=False,
                                            zca_whitening=False,
                                            rotation_range=0.,
                                            width_shift_range=0.,
                                            height_shift_range=0.,
                                            shear_range=0.,
                                            zoom_range=0.,
                                            channel_shift_range=0.,
                                            fill_mode='nearest',
                                            cval=0.,
                                            horizontal_flip=False,
                                            vertical_flip=False,
                                            rescale=None,
                                            preprocessing_function=None,
                                            data_format=K.image_data_format())
```

* **파라미터들**
  * **rotation_range** : 지정된 각도 범위 내에서 임의로 원본 이미지를 회전시킨다.
  * **width_shift_range** : 지정된 수평방향 이동 범위 내에서 임의로 원본 이미지를 이동시킨다.
  * **height_shift_range** : 지정된 수직방향 이동 범위 내에서 임의로 원본 이미지를 이동시킨다.
  * **shear_range** : 밀림 강도 범위 내에서 임의로 원본 이미지를 변형시킨다.
  * **zoom_range** : 지정된 확대/축소 범위 내에서 임의로 원본 이미지를 확대/축소 한다.
  * **horizontal_flip** : 수평방향으로 뒤집기
  * **vertical_flip** : 수직방향으로 뒤집기

<br>

### ImageDataGenerator() 함수를 이용한 코드

```python
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

# 랜덤시드 고정시키기
np.random.seed(5)

# 데이터셋 생성하기
data_aug_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=10,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.7,
                                  zoom_range=[0.9, 2.2],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

img = load_img('/content/drive/My Drive/Colab Notebooks/handwriting_shape/train/triangle/triangle001.png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

# 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복 횟수를 지정하며, 지정된 반복 횟수가 되면 빠져나오도록 해야 한다.
for batch in train_datagen.flow(x, batch_size=1, 
                                save_to_dir='/content/drive/My Drive/Colab Notebooks/handwriting_shape/preview', 
                                save_prefix='tri', save_format='png'):
  i += 1
  if i > 30:
    break
```