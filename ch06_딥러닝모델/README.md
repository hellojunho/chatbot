# 6장. 챗봇 엔진에 필요한 딥러닝 모델

## 6.1 케라스(Keras)
여러 딥러닝 프레임워크가 있지만, `케라스(Keras)`는 직관적이고 사용하기 쉽다는 장점이 있다.  
무엇보다 `빠른 개발`을 목적으로 두고, 모듈 구성이 간단하여 비전문가들도 상대적으로 쉽게 사용할 수 있다.  
`케라스`는 `신경망 모델`을 구축할 수 있는 `고수준 API 라이브러리`이다.  
최근에 `텐서플로우 2.0`에 기본 API로 채택되어 구글의 지원을 받고있다.  

### 6.1.1 인공 신경망
`인공 신경망`은 두뇌의 `뉴런`을 모방한 모델이다.  

입력: x0, x1, x2  
가중치 : w0, w1, w2  

뉴런 : ∑wi*xi + b | f  
출력 : f(∑wi*xi + b)

뉴런의 계산 과정을 보면 입력된 xi값들과 대응되는 뉴런의 가중치 wi값들을 각각 곱해서 모두 더해준다.  
그리고 `편향값 b`를 통해 결과값을 조정한다.  
이를 수학적으로 표현하면 간단한 1차 함수의 모양이다.  
```
y = (w0x0 + w1x1 + w2x2) + b
```
실제 뉴런은 특정 강도 이상일 때만 다음 뉴런으로 신호를 전달하는데, 인공 신경망에서는 뉴런의 처리 과정 중 `f`로 표시된 영역이다.  
이를 `활성화 함수`라고 하며, 가중치 계산 결과값 y가 최종적으로 어떤 형태의 출력값으로 내보낼지 결정한다.  
활성화 함수는 종류가 많은데, 여기서는 3가지만 다룬다.  
1. 스텝 함수
`스텝 함수`는 가장 기본이 되는 함수이다.  
그래프가 `계단`같이 생겨서 스텝 함수이고, 입력값이 0보다 크면 1로, 0 이하일 때는 0으로 만든다.  
즉 입력값이 양수일 때만 활성화 시킨다.  

2. 시그모이드 함수
이진 분류에서 스텝 함수의 문제를 해결하기 위해 사용한다.  
스텝 함수에서 판단의 기준이 되는 `임계치` 부근의 데이터를 고려하지 않는 문제를 해결하기 위해 계단 모양을 완만한 형태로 표현했다.  
수식 : S(t) = 1 / (1 + e^-t)

3. ReLU 함수
입력값이 0 이상인 경우에는 기울기가 1인 직선이고, 0보다 작을 때는 결과값이 0이다.  
`시그모이드 함수`에 비해 연산 비용이 크지 않아 학습 속도가 빠르다.  

실제로 문제를 신경망 모델로 해결할 때는 1개의 뉴런만 사용하는 것이 아니라, 문제가 복잡할 수록 뉴런의 수가 늘어나고 신경망의 계층도 깊어진다.  

입력층과 출력층으로만 구성되어 있는 단순한 신경망을 `단층 신경망`이라고 한다.  
입력층과 1개 이상의 은닉층, 출력층으로 구성되어 있는 신경망은 `심층 신경망`이라고 한다.  
흔히 `딥러닝`과 `신경망`이라고 부르는 것들이 바로 `심층 신경망`이다.  
*신경망 계층이 `깊게(deep)`구성되어 각각의 뉴런을 `학습(learning)`시킨다 해서 딥러닝*

주로 `입력층`을 구성하는 뉴런들은 1개의 입력값을 갖고, 가중치와 활성화 함수를 갖고 있지 않아 입력된 값을 그대로 출력하는 특징이 있다.  
`출력층`의 뉴런은 각각 1개의 출력값을 갖고 있고, 지정된 활성화 함수에 따른 출력 범위를 갖고 있다.  
"복잡한 문제일 수록 뉴런와 `은닉층` 수를 늘리면 좋다"라고 하지만, 계산해야 하는 파라미터가 많아지면 비용이 높아지는 단점이 있다.  

신경망에 대해서 알아보자.  
신경망 모델에서 입력층으로부터 출력층까지 데이터가 순방향으로 전파되는 과정을 `순전파`라고 한다.  
데이터가 순방형으로 전파될 때 현 단계 뉴런의 가중치와 전 단계 뉴런의 출력값의 곱을 입력값으로 받는다.  
이 값은 활성화 함수를 통해 다음 뉴런으로~  

[순전파 예시]  
입력층: x --(전파: w_h * x)--> 은닉층: h --(전파: w_y * h)--> 결과값(yout)  

결과값이 실제 값과 오차가 많다면?  
다음 순전파 진행 시 오차가 줄어드는 방향으로 가중치(w_h, w_y)를 역방향으로 갱신해나간다.  
이 과정을 `역전파`라고 한다.  

### 6.1.2 딥러닝 분류 모델 만들기
[History 객체]  
loss : 각 에포크마다의 학습 손실값  
accuracy : 각 에포크마다의 학습 정확도  
val_loss : 각 에포크마다의 검증 손실값  
val_accuracy : 각 에포크마다의 검증 정확도  
epoch(에포크) : 학습의 횟수  

```
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# MNIST 데이터셋 가져오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # 데이터 정규화

# tf.data를 사용하여 데이터셋을 섞고 배치 만들기
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
train_size = int(len(x_train) * 0.7) # 학슴셋 : 검증셋 = 7 : 3
train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).batch(20)

# MNIST 분류 모델 구성
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 생성
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.complie(loss='categorial_crossentropy', optimizer='sgd', metrics['accuracy'])

# 모델 학습
hist = model.fit(train_ds, validation_data=val_ds, epochs=10)

# 모델 평가
print('모델 평가')
model.evaluate(x_test, y_test)

# 모델 정보 출력
model.summary()

# 모델 저장
model.save('mnist_model.h5')

# 학습 결과 그래프 그리기
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['var_loss'], 'r', label='var loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()
```  
위 코드를 돌리다가 만약 주피터에서 이런 오류가 뜬다면??
> The kernel appears to have died. It will restart automatically.

이 오류가 발생하는 이유는 `커널이 죽어서`인데, 커널이 죽는 이유는 `메모리 할당량을 초과`했기 때문!  
[해결방법]  
1. `jupyter_notebook_config.py`에서 다음 코드를 입력하고 재부팅하기.    
    - 위의 .py파일은 `./jupyter`폴더 안에 있음  
> c.NotebookApp.max_buffer_size = 100000000000000000000000

이렇게 큰 값을 입력해 바꾸면 됨



## 6.2.1 CNN 모델 개념












