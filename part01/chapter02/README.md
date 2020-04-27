# CHAPTER 02. 맥에서 케라스 설치하기

## 가상 개발환경 만들기

프로젝트별로 개발환경이 다양하기 때문에 가상환경을 이용하면 편리하다.

1. **가상환경을 제공하는 virtualenv을 먼저 설치한다.**

   ```bash
   keras $ sudo pip install virtualenv
   ```

2. **실제 가상환경을 만든다.**

   ```bash
   keras $ virtualenv venv
   ```

3. **가상환경을 실행한다.**

   ```bash
   keras $ source venv/bin/activate
   ```

<br>

## 웹 기반 파이썬 개발환경인 주피터 노트북 설치

1. **주피터 노트북을 설치한다.**

   ```bash
   (venv) keras $ pip install ipython[notebook]
   ```

   * "Your pip version is out of date, ..." 이라는 에러가 발생하면 pip 업그레이드

     ```bash
     (venv) keras $ pip install --upgrade pip
     (venv) keras $ pip install ipython[notebook]
     ```

2. **주피터 노트북를 실행한다.**

   ```bash
   (venv) keras $ jupyter notebook
   ```

<br>

## 주요 패키지 설치

```bash
(venv) keras $ pip install numpy scipy scikit-learn matplotlib pandas pydot h5py
```

* pydot은 모델을 가시화할 때 필요한 것인데, 이를 사용하려면 graphviz가 필요하다.

  ```bash
  (venv) keras $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  (venv) keras $ brew install graphviz
  ```

<br>

## 딥러닝 라이브러리 설치

```bash
(venv) keras $ pip install theano tensorflow keras
```

<br>

## 설치환경 테스트해보기

1. 라이브러리 테스트

   ```python
   import scipy
   import numpy
   import matplotlib
   import pandas
   import sklearn
   import pydot
   import h5py
   
   import theano
   import tensorflow
   import keras
   
   print('scipy ' + scipy.__version__)
   print('numpy ' + numpy.__version__)
   print('matplotlib ' + matplotlib.__version__)
   print('pandas ' + pandas.__version__)
   print('sklearn ' + sklearn.__version__)
   print('pydot ' + pydot.__version__)
   print('h5py ' + h5py.__version__)
   
   print('theano ' + theano.__version__)
   print('tensorflow ' + tensorflow.__version__)
   print('keras ' + keras.__version__)
   ```

   > 각 패키지별로 버전이 표시되면 정상적으로 설치가 된 것이다.

2. 딥러닝 기본 모델 구동 확인

   ```python
   from keras.utils import np_utils
   from keras.datasets import mnist
   from keras.models import Sequential
   from keras.layers import Dense, Activation
   
   (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
   X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
   X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
   Y_train = np_utils.to_categorical(Y_train)
   Y_test = np_utils.to_categorical(Y_test)
   
   model = Sequential()
   model.add(Dense(units=64, input_dim=28*28, activation='relu'))
   model.add(Dense(units=10, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
   model.fit(X_train, Y_train, epochs=5, batch_size=32)
   
   loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
   
   print('loss_and_metrics : ' + str(loss_and_metrics))
   ```

   **실행 결과**

   ```
   Epoch 1/5
   60000/60000 [==============================] - 1s - loss: 0.6558 - acc: 0.8333     
   Epoch 2/5
   60000/60000 [==============================] - 1s - loss: 0.3485 - acc: 0.9012     
   Epoch 3/5
   60000/60000 [==============================] - 1s - loss: 0.3037 - acc: 0.9143     
   Epoch 4/5
   60000/60000 [==============================] - 1s - loss: 0.2759 - acc: 0.9222     
   Epoch 5/5
   60000/60000 [==============================] - 1s - loss: 0.2544 - acc: 0.9281     
    8064/10000 [=======================>......] - ETA: 0sloss_and_metrics : [0.23770418465733528, 0.93089999999999995]
   ```

   > 에러 없이 화면이 출력되면 정상적으로 작동되는 것이다.

3. 딥러닝 모델 가시화 기능 확인

   ```python
   from IPython.display import SVG
   from keras.utils.vis_utils import model_to_dot
   
   %matplotlib inline
   
   SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
   ```

   <img src="http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Mac_4.png">

4. 딥러닝 모델 저장 기능 확인

   ```python
   from keras.models import load_model
   
   model.save('mnist_mlp_model.h5')
   model = load_model('mnist_mlp_model.h5')
   ```

<br>

## 오류 대처

#### 주피터 실행 에러

> jupyter notebook를 실행하면, “Open location 메시지를 인식할수 없습니다. (-1708)” 또는 “execution error: doesn’t understand the “open location” message. (-1708)” 메시지가 뜹니다.

운영체제 버전 등의 문제로 주피터가 실행할 브라우저를 찾지 못하는 경우 발생하는 메시지입니다. 이 경우 주피터 옵션에 브라우저를 직접 셋팅하시면 됩니다. ‘.jupyter_notebook_config.py’ 파일이 있는 지 확인합니다.

```
(venv) keras_talk $ find ~/.jupyter -name jupyter_notebook_config.py
```

출력되는 내용이 없다면 파일이 없는 것입니다. 파일이 없다면 아래 명령으로 파일을 생성합니다.

```
(venv) keras_talk $ jupyter notebook --generate-config 
```

‘jupyter_notebook_config.py’파일을 엽니다.

```
(venv) keras_talk $ vi ~/.jupyter/jupyter_notebook_config.py
```

아래와 같이 ‘c.Notebook.App.browser’변수를 찾습니다.

```
# If not specified, the default browser will be determined by the `webbrowser`
# standard library module, which allows setting of the BROWSER environment
# variable to override it.
# c.NotebookApp.browser = u''
```

‘c.NotebookApp.browser’ 변수를 원하는 브러우저 이름으로 설정합니다. 아래 행 중 하나만 설정하시고, 앞에 ‘#’은 제거해야 합니다.

```
c.NotebookApp.browser = u’chrome’
c.NotebookApp.browser = u’safari’
c.NotebookApp.browser = u’firefox’
```

이 파일을 저장 후 (esc키 누른 후 wq! 입력하고 엔터칩니다.) 다시 주피터를 실행하면 지정한 브라우저에서 정상적으로 실행되는 것을 볼 수 있습니다. 설정한 이후에도 해당 브라우저의 경로가 설정되어 있지 않다면 아래와 같은 오류가 발생합니다.

```
No web browser found: could not locate runnable browser.
```

이 경우 해당 브러우저의 전체 경로를 설정합니다.

```
c.NotebookApp.browser = u'open -a /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome %s'
```

