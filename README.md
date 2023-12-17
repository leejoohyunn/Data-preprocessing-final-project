# Data-preprocessing-final-project
# 하이퍼파라미터 튜닝
## 1.1 하이퍼파라미터의 역할과 중요성
### 1.1.1 하이퍼파라미터와 모델 매개변수의 차이 (Hyperparameter vs. Parameter)
- 파라미터(매개변수): 모델이 데이터에서 규칙을 학습하는 데 사용되는 변수이며 훈련 과정에서 알고리즘에 의해 업데이트 됩니다. 파라미터에 대해 최적의 값을 설정하지 않지만, 데이터에서 자체 값을 학습합니다. 파라미터의 최적 값이 찾아지면 모델은 훈련을 마칩니다.
- 하이퍼파라미터(초매개변수): 모델 훈련을 제어하는 변수입니다. 따라서 하이퍼파라미터는 파라미터의 값을 제어할 수 있습니다. 즉, 파라미터의 최적 값은 사용하는 하이퍼파라미터의 값에 따라 달라집니다. 파라미터와는 달리, 하이퍼파라미터는 데이터에서 값을 학습하지 않고, 모델을 훈련하기 전에 수동으로 지정해야 합니다. 일단 지정되면 하이퍼파라미터 값은 모델 훈련 중에 고정됩니다. 
(https://medium.com/data-science-365/parameters-vs-hyperparameters-what-is-the-difference-5f40e16e2e82)
(https://velog.io/@emseoyk/%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D)

| 하이퍼파리미터                             | 파라미터                                          |
|------------------------------------------:|--------------------------------------------------:|
|학습 과정에 반영되는 값, 학습 시작 전 미리 조정| 데이터로부터 학습 또는 예측되는 값, 모델 내부에서 결정|
|ex) 학습률, 손실 함수, 배치 사이즈            |선형회귀 계수, 가중치, 편향, 평균, 표준편차           |
|직접 조정 가능                               |직접 조정 불가능                                    |

## 1.2 하이퍼파라미터의 종류
### 1. **학습률**:


: 모델이 한 번의 학습 단계에서 얼마나 많이 학습하는지를 조절합니다. 혹은, 손실함수(정답과 예측값 차이를 계산한 함수)를 최소화하는 파라미터를 찾는것이라고 할 수 있습니다. 아래 그림 같은 손실함수의 포물선에서 경사를 따라 이동하는 양입니다. 너무 작은 학습률은 수렴을 느리게 할 수 있고, 너무 큰 학습률은 발산의 위험을 가지고 있습니다.
![학습률 이미지](https://github.com/leejoohyunn/images/blob/main/image.png)


![학습률 이미지](https://github.com/leejoohyunn/images/blob/main/img.png)

```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))  # 2차원 배열로 초기화
        self.bias = 0

        for _ in range(self.n_iterations):
            # 예측값 계산
            y_pred = np.dot(X, self.weights) + self.bias

            # 경사 하강법 업데이트
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 데이터 생성 및 분할
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 학습률 리스트
learning_rates = [0.1, 0.01, 0.001]

# 결과 저장을 위한 딕셔너리
results = {}

# 각 학습률에 대해 모델 학습 및 평가
for lr in learning_rates:
    model = LinearRegressionGradientDescent(learning_rate=lr, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # 결과 저장
    results[lr] = {'model': model, 'mse': mse}

# 결과 출력
for lr, result in results.items():
    print(f"Learning Rate: {lr}, MSE: {result['mse']}")

# 결과 시각화
plt.figure(figsize=(10, 6))
for lr, result in results.items():
    plt.plot(X_test, result['model'].predict(X_test), label=f'LR={lr}')

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.title('Linear Regression with Different Learning Rates')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```
```python
Learning Rate: 0.1, MSE: 0.6536995222280376
Learning Rate: 0.01, MSE: 0.6926651409345591
Learning Rate: 0.001, MSE: 2.0816606850501587
```
![결과 이미지](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(1).png)


### 2. **배치 크기(batch size)**: 모델이 각 학습 단계에서 처리하는 데이터 샘플의 개수를 나타냅니다. 적절한 배치 크기를 선택하는 것은 모델의 효율성에 영향을 미칩니다. 최적화를 진행하다가 극소값 혹은 안정점에 빠질 수 도 있어 크기가 크면 무조건 빠르고 효과적으로 최적화가 이뤄지는 것은 아닙니다. 배치 크기가 크면, 데이터의 평균적인 특성을 바탕으로 학습이 진행돼 gradient(기울기)를 크게 바꾸지 못합니다. 다시 말하면, 평균이 구해지면 특이값이 묻혀 영향력이 작아집니다. 반대로, 배치 사이즈가 작을 때는 이 구간을 빠져나오기 비교적 수월하다. 배치 사이즈는 학습 속도와 학습 성능에 모두 영향을 미치는 중요한 요소입니다. 이를 고려해 최적의 배치 사이즈를 부여해야합니다. 배치 크기는 보통 2의 제곱수를 사용합니다. CPU와 GPU의 메모리가 2의 배수여서 2의 제곱 수 일 경우에 데이터 송수신의 효율을 높일 수 있습니
   
![배치 사이즈 이미지](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.png
)
![배치 사이즈 이미지](https://github.com/leejoohyunn/images/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202023-12-17%20160749.png)
https://wikidocs.net/55580

> **미니 배치**
> 미니 배치란, 전체 데이터를 N등분해 각각의 학습 데이터를 배치 방식으로 학습시킨다. 즉, 전체 데이터 세을 몇 개의 데이터셋으로 나누었을 때, 그 작은 데이터 셋의 뭉치입니다. 미니 배치를 사용하는 이유는 데이터가 많을 때, 길어지는 시간이나 데이터의 손실을 줄이기 위해서 입니다.
https://welcome-to-dewy-world.tistory.com/86

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionMiniBatchFixedLR:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            # 미니배치 샘플 선택
            indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            # 예측값 계산
            y_pred = np.dot(X_batch, self.weights) + self.bias

            # 경사 하강법 업데이트
            dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1/self.batch_size) * np.sum(y_pred - y_batch)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 데이터 생성 및 분할
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 고정된 학습률
learning_rate = 0.01

# 미니배치 사이즈 리스트
batch_sizes = [16, 32, 64]

# 결과 저장을 위한 딕셔너리
results = {}

# 각 미니배치 사이즈에 대해 모델 학습 및 평가
for batch_size in batch_sizes:
    model = LinearRegressionMiniBatchFixedLR(learning_rate=learning_rate, n_iterations=1000, batch_size=batch_size)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 결과 저장
    results[batch_size] = {'model': model, 'mse': mse}

# 결과 출력
for batch_size, result in results.items():
    print(f"Batch Size: {batch_size}, MSE: {result['mse']}")

# 결과 시각화
plt.figure(figsize=(12, 8))
for batch_size, result in results.items():
    plt.plot(X_test, result['model'].predict(X_test), label=f'Batch Size={batch_size}')

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.title('Linear Regression with Fixed Learning Rate and Different Batch Sizes')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```
```python
Batch Size: 16, MSE: 0.690311253484263
Batch Size: 32, MSE: 0.6927622099151095
Batch Size: 64, MSE: 0.6949990964496909
```
![배치 사이즈 이미지](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(2).png)

### 3. **에포크 수(Number of Epochs)**: 전체 데이터셋을 한 번 훈련하는 것을 1 에포크라고 합니다. 에포크 수는 전체 데이터셋을 몇 번 반복해서 훈련할지를 결정합니다. 에포크 수를 높일수록, 다양한 무작위 가중치를 학습하는 것으로, 적합한 파라미터를 찾을 확률이 올라갑니다. 하지만, 에프크를 지나치게 높일 경우, 학습 데이터가 과적합되어 다른 데이터를 적용했을 때 제대로된 예측을 못합니다. 에포크는 학습 데이터셋 샘플의 수와 동일하게 하며 이는 배치수와 배치 사이즈를 곱한 값과 같습니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionFixedParams:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            # 미니배치 샘플 선택
            indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            # 예측값 계산
            y_pred = np.dot(X_batch, self.weights) + self.bias

            # 경사 하강법 업데이트
            dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1/self.batch_size) * np.sum(y_pred - y_batch)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 데이터 생성 및 분할
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 고정된 학습률 및 미니배치 사이즈
learning_rate = 0.01
batch_size = 32

# 에포크 수 리스트
n_iterations_list = [100, 500, 1000, 2000]

# 결과 저장을 위한 딕셔너리
results = {}

# 각 에포크 수에 대해 모델 학습 및 평가
for n_iterations in n_iterations_list:
    model = LinearRegressionFixedParams(learning_rate=learning_rate, n_iterations=n_iterations, batch_size=batch_size)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 결과 저장
    results[n_iterations] = {'model': model, 'mse': mse}

# 결과 출력
for n_iterations, result in results.items():
    print(f"Epochs: {n_iterations}, MSE: {result['mse']}")

# 결과 시각화
plt.figure(figsize=(12, 8))
for n_iterations, result in results.items():
    plt.plot(X_test, result['model'].predict(X_test), label=f'Epochs={n_iterations}')

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.title('Linear Regression with Fixed Learning Rate and Batch Size, Different Epochs')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

```
```python
Epochs: 100, MSE: 2.078373541112038
Epochs: 500, MSE: 0.782228324116659
Epochs: 1000, MSE: 0.6894744940599438
Epochs: 2000, MSE: 0.661203028560142
```
![에포크](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(3).png)

### 4. **가중치 감소(Weight Decay)**: 과적합을 방지하기 위해 가중치 감소를 사용합니다. 이는 가중치 값이 너무 크지 않도록 제한하는 역할을 합니다. 가중치 감소에는 규제(Regularization)이 이용된다. Regularization 이란 wieght의 절대값을 작게 만들며, weight의 모든 원소를 0에 가깝게해 특성이 출력에 주는 영향을 최소화로 만든다. 즉, 오버피팅되지 않도록 모델을 제한한다는 것이다. 대표적인 Regularization으로는 L1과 L2가 있다. 

>**L1 규제**: 각 weight의 제곱합에 규제 강도(Regularization) λ를 곱하고 그 값을 loss function(손실함수)에 더한다. λ를 크게 하면 가중치가 감소되고, λ를 작게하면 가중치가 증가한다. 일반적으로 L2 규제가 많이 쓰인다. 

 >**L2 규제**: weight의 제곱의 합이 아닌 가중치 합을 더해 regualrization strength λ를 곱해 오차에 더한다. L2 규제와 달리 L1 규제할 때는 일부 가중치 값이 0이된다. 이를 통해 모델에 대한 이해도가 높아지고, 모델에서 중요한 feature이 무엇인지 알 수 있다.
>https://goatlab.tistory.com/124

![배치 사이즈 이미지](https://github.com/leejoohyunn/images/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202023-12-17%20170336.png)
https://sacko.tistory.com/45

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionWeightDecay:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            # 미니배치 샘플 선택
            indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            # 예측값 계산
            y_pred = np.dot(X_batch, self.weights) + self.bias

            # 경사 하강법 업데이트 (가중치 감소 포함)
            dw = (1/self.batch_size) * (np.dot(X_batch.T, (y_pred - y_batch)) + 2 * self.weight_decay * self.weights)
            db = (1/self.batch_size) * np.sum(y_pred - y_batch)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 데이터 생성 및 분할
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 고정된 학습률 및 미니배치 사이즈
learning_rate = 0.01
batch_size = 32

# 가중치 감소 값 리스트
weight_decay_values = [0.0, 0.01, 0.1, 1.0]

# 결과 저장을 위한 딕셔너리
results = {}

# 각 가중치 감소 값에 대해 모델 학습 및 평가
for weight_decay in weight_decay_values:
    model = LinearRegressionWeightDecay(learning_rate=learning_rate, n_iterations=1000, batch_size=batch_size, weight_decay=weight_decay)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 결과 저장
    results[weight_decay] = {'model': model, 'mse': mse}

# 결과 출력
for weight_decay, result in results.items():
    print(f"Weight Decay: {weight_decay}, MSE: {result['mse']}")

# 결과 시각화
plt.figure(figsize=(12, 8))
for weight_decay, result in results.items():
    plt.plot(X_test, result['model'].predict(X_test), label=f'Weight Decay={weight_decay}')

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.title('Linear Regression with Fixed Learning Rate and Batch Size, Different Weight Decay')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

```
```python
Weight Decay: 0.0, MSE: 0.6899889858180998
Weight Decay: 0.01, MSE: 0.6918531438504338
Weight Decay: 0.1, MSE: 0.6850834668730379
Weight Decay: 1.0, MSE: 0.6609202933086811
```
![가중치 감소 결과](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(4).png)


### 5. **드롭아웃 비율(Dropout Rate)**:드롭아웃은 학습 중에 무작위로 일부 뉴런을 제외하여 모델의 일반화 성능을 향상시키는 데 사용됩니다. 드롭아웃 비율은 제외될 뉴런의 비율을 나타냅니다.일반적으로 학습할 때만 드롭아웃을 사용하고, 예측시에는 사용하지 않는다. 학습할 때 인공 신경망이 특정 뉴런 또는 특정 조합에 의존적이게 되는것을 방지해준다.

![배치 사이즈 이미지](https://github.com/leejoohyunn/images/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202023-12-17%20170445.png)

>**앙상블 기법(Ensemble)**:
>여러 모델을 종합적으로 고려해 최적의 결과를 찾는것이다. 학습할 수 있는 장비가 많을 때 사용하는 방법으로, 다수의 돗립적인 학습 모델을 만들어 각자 학습한 뒤 모델들을 합쳐 한 번의 예측을 만드는것입니다. 이러한 점에서 앙상블은 드롭아웃과 유사하다는 것을 알 수 있습니다. 
https://childult-programmer.tistory.com/44
>
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionDropout:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32, dropout_rate=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            # 미니배치 샘플 선택
            indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            # 예측값 계산
            y_pred = np.dot(X_batch, self.weights) + self.bias

            # 드롭아웃 적용
            mask = np.random.rand(*X_batch.shape) > self.dropout_rate
            X_batch *= mask

            # 경사 하강법 업데이트
            dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1/self.batch_size) * np.sum(y_pred - y_batch)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 데이터 생성 및 분할
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 고정된 학습률 및 미니배치 사이즈
learning_rate = 0.01
batch_size = 32

# 드롭아웃 값 리스트
dropout_rates = [0.0, 0.2, 0.5, 0.8]

# 결과 저장을 위한 딕셔너리
results = {}

# 각 드롭아웃 값에 대해 모델 학습 및 평가
for dropout_rate in dropout_rates:
    model = LinearRegressionDropout(learning_rate=learning_rate, n_iterations=1000, batch_size=batch_size, dropout_rate=dropout_rate)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 결과 저장
    results[dropout_rate] = {'model': model, 'mse': mse}

# 결과 출력
for dropout_rate, result in results.items():
    print(f"Dropout Rate: {dropout_rate}, MSE: {result['mse']}")

# 결과 시각화
plt.figure(figsize=(12, 8))
for dropout_rate, result in results.items():
    plt.plot(X_test, result['model'].predict(X_test), label=f'Dropout Rate={dropout_rate}')

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.title('Linear Regression with Fixed Learning Rate and Batch Size, Different Dropout Rates')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

```
```python
Dropout Rate: 0.0, MSE: 0.6991444738804115
Dropout Rate: 0.2, MSE: 0.6753189195400063
Dropout Rate: 0.5, MSE: 0.6403026506559619
Dropout Rate: 0.8, MSE: 0.8668981023338229
```
![드롭아웃 결과](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(5).png)


### 6. **활성화 함수(Activation Function)**:딥러닝 네트워크에서 노드에 입력된 값들을 비선형 함수에 통과시킨 후 다음 레이어로 전달하는데, 이때 활성화함수를 사용한다. 비선형 함수를 사용하는 이유는 딥러닝 모델의 레이어 층을 깊게 구성할 수 있기 때문이다. 활성화 함수의 종류로는 sigmoid 함수, Tanh 함수, ReLU 함수, Leaky ReLU, PReLU, ELU, Maxout 등이 있다.
>**시그모이드 함수**:
>
>특징:
   범위: (0, 1)
   출력이 0 또는 1에 가까워지면 그래디언트 소실 문제 발생 가능.
>
>사용:
   이진 분류 문제의 출력층에서 주로 사용.

>**Tanh 함수**:
>
>특징:
   범위: (-1, 1)
   시그모이드와 유사하지만 출력 범위가 더 넓어 그래디언트 소실 문제가 상대적으로 줄어듦.
>
>사용:
   이진 분류 문제나 RNN(순환 신경망)에서 활성화 함수로 사용.

>**ReLU**:
>
>특징:
   양수 입력에 대해 선형, 음수 입력에 대해 0을 출력.
   학습이 빠르고 계산 효율성이 뛰어남.
>
>사용:
   컨볼루션 신경망 (CNN) 및 이미지 인식과 같은 분야에서 주로 사용.
   주의: 입력이 음수인 경우, 그래디언트 소실 문제가 발생할 수 있음.

>**Leaky ReLU**:
>
>특징:
   음수 입력에 대해 작은 기울기를 가진 선형 함수.
   ReLU의 문제를 해결하기 위해 도입.
>
>사용:
   일반적으로 ReLU의 대안으로 사용.

>**PReLU**:
>
>특징:
   Leaky ReLU의 확장으로 음수 입력에 대해 학습 가능한 기울기를 가짐.

>사용:
   데이터셋에 따라 Leaky ReLU보다 더 나은 성능을 보일 수 있음.

>**ELU**:
>
>특징:
   음수 입력에 대해 작은 기울기를 가진 지수 함수.
   ReLU의 문제를 완화하면서 그래디언트 소실 문제를 줄임.
>
>사용:
   신경망의 히든 레이어에서 성능이 좋을 수 있음.

>**Maxout**:
>
>특징:
   두 개의 입력 중 더 큰 값을 선택.
   매개변수가 더 많아 계산 비용이 높을 수 있음.
>
>사용:
   매우 깊은 네트워크에서 특히 성능 향상을 위해 사용.
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=handuelly&logNo=221824080339
![활성화함수 이미지](https://github.com/leejoohyunn/images/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202023-12-17%20181409.png)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionActivation:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32, activation_function='linear'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.activation_function = activation_function
        self.weights = None
        self.bias = None

    def linear(self, x):
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def prelu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        activation_function = getattr(self, self.activation_function, self.linear)

        for _ in range(self.n_iterations):
            # 미니배치 샘플 선택
            indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            # 예측값 계산 및 활성화 함수 적용
            y_pred = activation_function(np.dot(X_batch, self.weights) + self.bias)

            # 경사 하강법 업데이트
            dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1/self.batch_size) * np.sum(y_pred - y_batch)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        activation_function = getattr(self, self.activation_function, self.linear)
        return activation_function(np.dot(X, self.weights) + self.bias)

# 데이터 생성 및 분할
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 고정된 학습률, 미니배치 사이즈 및 다양한 활성화 함수
learning_rate = 0.01
batch_size = 32
activation_functions = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'prelu', 'elu']

# 결과 저장을 위한 딕셔너리
results = {}

# 각 활성화 함수에 대해 모델 학습 및 평가
for activation_function in activation_functions:
    model = LinearRegressionActivation(learning_rate=learning_rate, n_iterations=1000, batch_size=batch_size, activation_function=activation_function)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 결과 저장
    results[activation_function] = {'model': model, 'mse': mse}

# 결과 출력
for activation_function, result in results.items():
    print(f"Activation Function: {activation_function}, MSE: {result['mse']}")

# 결과 시각화
plt.figure(figsize=(12, 8))
for activation_function, result in results.items():
    plt.plot(X_test, result['model'].predict(X_test), label=f'Activation Function={activation_function}')

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.title('Linear Regression with Fixed Learning Rate and Batch Size, Different Activation Functions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```
```python
Activation Function: linear, MSE: 0.6899889858180998
Activation Function: sigmoid, MSE: 39.93863846944045
Activation Function: tanh, MSE: 39.93863846944045
Activation Function: relu, MSE: 0.6857353119803922
Activation Function: leaky_relu, MSE: 0.6872081168858777
Activation Function: prelu, MSE: 0.6953053685875947
Activation Function: elu, MSE: 0.6943654170950561
```
![활성화 함수 임지](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(6).png)


### 7. **최적화 알고리즘(Optimizer)**: 딥러닝 학습시 손실함수의 최솟값을 찾아가는 것을 최적화(Optimization)이라고 하며, 이를 수행하는 알고리즘이 최적화 알고리즘(Optimizer)이다. 모델의 가중치를 업데이트하는 데 사용되는 최적화 알고리즘을 선택합니다. 대표적으로는 SGD, Adam, RMSprop, Adagrad,NAG, Momentum 등이 있습니다.

>**SGD(Stochastic Gradient Descent 확률적 경사하강법)**:
>
>가장 기본적인 최적화 알고리즘.
각 배치에서 그래디언트를 계산하고 가중치를 업데이트.
단점: 수렴이 느리고 지그재그 움직임이 있을 수 있음.

>**Adam(Adaptive Moment Estimation)**:
>
>그래디언트의 지수 이동 평균을 사용하여 적응적으로 학습률을 조정.
AdaGrad 및 RMSprop의 아이디어를 결합함.
다양한 문제에서 효과적으로 사용되며, 기본적으로 많이 사용되는 알고리즘 중 하나.

>**RMSprop(Root Mean Square Propagation)**:
>
>과거의 제곱 그래디언트에 기반하여 학습률을 조정.
각 매개변수에 대해 개별적으로 학습률을 조절함으로써 SGD의 단점을 보완.
https://velog.io/@freesky/Optimizer

>**Adagrad(Adaptive Gradient Algorithm)**:
>
>각 매개변수에 대해 학습률을 조정.
자주 등장하지 않는 특성에 대한 학습률을 증가시켜 적응적으로 학습.

>**NAG(Nesterov Accelerated Gradient)**:
>
>모멘텀 방법에서 그래디언트의 지연된 버전을 사용하여 가중치 업데이트.
모멘텀 방법에 비해 수렴이 더 빠르고 안정적.

>**Momentum**:
>
>과거의 그래디언트 정보를 사용하여 가중치 업데이트.
기존 SGD에 비해 더 빠른 수렴을 도모하며, 지역 최소값에서 덜 갇히는 경향.
https://heeya-stupidbutstudying.tistory.com/entry/ML-%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%97%90%EC%84%9C%EC%9D%98-Optimizer-%EC%97%AD%ED%95%A0%EA%B3%BC-%EC%A2%85%EB%A5%98
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionOptimizer:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32, optimizer='sgd'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.weights = None
        self.bias = None
        self.velocity = None

    def sgd_update(self, dw, db):
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def adam_update(self, dw, db, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if self.velocity is None:
            self.velocity = {'dw': 0, 'db': 0}
            self.momentum = {'dw': 0, 'db': 0}
            self.t = 0

        self.t += 1
        self.momentum['dw'] = beta1 * self.momentum['dw'] + (1 - beta1) * dw
        self.momentum['db'] = beta1 * self.momentum['db'] + (1 - beta1) * db

        self.velocity['dw'] = beta2 * self.velocity['dw'] + (1 - beta2) * (dw**2)
        self.velocity['db'] = beta2 * self.velocity['db'] + (1 - beta2) * (db**2)

        m_dw_hat = self.momentum['dw'] / (1 - beta1**self.t)
        m_db_hat = self.momentum['db'] / (1 - beta1**self.t)

        v_dw_hat = self.velocity['dw'] / (1 - beta2**self.t)
        v_db_hat = self.velocity['db'] / (1 - beta2**self.t)

        self.weights -= self.learning_rate * m_dw_hat / (np.sqrt(v_dw_hat) + epsilon)
        self.bias -= self.learning_rate * m_db_hat / (np.sqrt(v_db_hat) + epsilon)

    def rmsprop_update(self, dw, db, gamma=0.9, epsilon=1e-8):
        if self.velocity is None:
            self.velocity = {'dw': 0, 'db': 0}

        self.velocity['dw'] = gamma * self.velocity['dw'] + (1 - gamma) * (dw**2)
        self.velocity['db'] = gamma * self.velocity['db'] + (1 - gamma) * (db**2)

        self.weights -= self.learning_rate * dw / (np.sqrt(self.velocity['dw']) + epsilon)
        self.bias -= self.learning_rate * db / (np.sqrt(self.velocity['db']) + epsilon)

    def adagrad_update(self, dw, db, epsilon=1e-8):
        if self.velocity is None:
            self.velocity = {'dw': 0, 'db': 0}

        self.velocity['dw'] += dw**2
        self.velocity['db'] += db**2

        self.weights -= self.learning_rate * dw / (np.sqrt(self.velocity['dw']) + epsilon)
        self.bias -= self.learning_rate * db / (np.sqrt(self.velocity['db']) + epsilon)

    def nag_update(self, dw, db, mu=0.9):
        if self.velocity is None:
            self.velocity = {'dw': 0, 'db': 0}

        self.velocity['dw'] = mu * self.velocity['dw'] - self.learning_rate * dw
        self.velocity['db'] = mu * self.velocity['db'] - self.learning_rate * db

        self.weights += self.velocity['dw']
        self.bias += self.velocity['db']

    def momentum_update(self, dw, db, mu=0.9):
        if self.velocity is None:
            self.velocity = {'dw': 0, 'db': 0}

        self.velocity['dw'] = mu * self.velocity['dw'] - self.learning_rate * dw
        self.velocity['db'] = mu * self.velocity['db'] - self.learning_rate * db

        self.weights += self.velocity['dw']
        self.bias += self.velocity['db']

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            # 미니배치 샘플 선택
            indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            # 예측값 계산
            y_pred = np.dot(X_batch, self.weights) + self.bias

            # 경사 하강법 업데이트
            dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1/self.batch_size) * np.sum(y_pred - y_batch)

            # 옵티마이저에 따라 업데이트
            if self.optimizer == 'sgd':
                self.sgd_update(dw, db)
            elif self.optimizer == 'adam':
                self.adam_update(dw, db)
            elif self.optimizer == 'rmsprop':
                self.rmsprop_update(dw, db)
            elif self.optimizer == 'adagrad':
                self.adagrad_update(dw, db)
            elif self.optimizer == 'nag':
                self.nag_update(dw, db)
            elif self.optimizer == 'momentum':
                self.momentum_update(dw, db)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 데이터 생성 및 분할
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 고정된 학습률, 미니배치 사이즈 및 다양한 옵티마이저
learning_rate = 0.01
batch_size = 32
optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad', 'nag', 'momentum']

# 결과 저장을 위한 딕셔너리
results = {}

# 각 옵티마이저에 대해 모델 학습 및 평가
for optimizer in optimizers:
    model = LinearRegressionOptimizer(learning_rate=learning_rate, n_iterations=1000, batch_size=batch_size, optimizer=optimizer)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 결과 저장
    results[optimizer] = {'model': model, 'mse': mse}

# 결과 출력
for optimizer, result in results.items():
    print(f"Optimizer: {optimizer}, MSE: {result['mse']}")

# 결과 시각화
plt.figure(figsize=(12, 8))
for optimizer, result in results.items():
    plt.plot(X_test, result['model'].predict(X_test), label=f'Optimizer={optimizer}')

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.title('Linear Regression with Fixed Learning Rate and Batch Size, Different Optimizers')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```
```python
Optimizer: sgd, MSE: 0.6899889858180998
Optimizer: adam, MSE: 0.8153083663743448
Optimizer: rmsprop, MSE: 0.6450898835713961
Optimizer: adagrad, MSE: 36.80737819137021
Optimizer: nag, MSE: 0.6625120212608344
Optimizer: momentum, MSE: 0.6856392345679706
```
![optimizer result](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(7).png)


### 8. **은닉층 수와 뉴런 수**:신경망의 구조를 결정하는 하이퍼파라미터로, 은닉층의 수와 각 은닉층의 뉴런 수를 조절합니다. 일반적으로 다층 퍼셉트론에선 3~5개의 은닉층을 쌓고, CNN을 통한 이미지 처리에는많게는 1000개의 은닉층이 쌓이기도 한다. 하지만 은닉층이 증가하면 기울기 소실(Gradient Vanishing)이 발생하기도 한다
>Gradient Vanishing 현상을 방지하기 위한 방법으로는
>
>1. ReLU와 ReLU의 변형들을 사용하는 것입니다. 시그모이드나 하이퍼볼릭탄젠트 대신 사용합니다.
>2. 가중치 초기화(Weight initialization) 가령, He 초기화나 Xavier 초기화가 있습니다
>3. 배치 정규화. 각 층에 입력을 평균과 분산으로 정규화해 학습을 효율적으로 합니다. 하지만, 배치 정규화는 미니 배치 크기에 의존적이며 RNN에 적용하기 어렵다는 점이 한계입니다.
>4. 층 정규화. 배치 정규화는 층간 정규화를 진행했다면 층 정규화는 층내 정규화를 진행하는것입니다.
>5. https://wikidocs.net/61271
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

class NeuralNetwork:
    def __init__(self, hidden_layer_sizes=(10,), learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights, self.bias = self.initialize_parameters()

    def initialize_parameters(self):
        hidden_layer_sizes = [1] + list(self.hidden_layer_sizes) + [1]
        weights = [np.random.randn(hidden_layer_sizes[i], hidden_layer_sizes[i+1]) for i in range(len(hidden_layer_sizes)-1)]
        bias = [np.zeros((1, hidden_layer_sizes[i+1])) for i in range(len(hidden_layer_sizes)-1)]
        return weights, bias

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward_pass(self, X, weights, bias):
        activations = [X]
        for w, b in zip(weights[:-1], bias[:-1]):
            activations.append(self.relu(np.dot(activations[-1], w) + b))
        activations.append(np.dot(activations[-1], weights[-1]) + bias[-1])
        return activations

    def compute_loss(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def backward_pass(self, X, y, activations, weights, bias):
        n_samples = X.shape[0]
        gradients = []

        # 역전파
        error = y - activations[-1]
        gradients.append(error)

        for i in range(len(weights)-2, -1, -1):
            error = gradients[-1].dot(weights[i+1].T) * self.relu_derivative(activations[i+1])
            gradients.append(error)

        gradients.reverse()

        # 가중치 및 편향 업데이트
        for i in range(len(weights)):
            weights[i] += self.learning_rate * activations[i].T.dot(gradients[i]) / n_samples
            bias[i] += self.learning_rate * np.sum(gradients[i], axis=0, keepdims=True) / n_samples

        return gradients

    def fit(self, X, y):
        weights, bias = self.initialize_parameters()
        for _ in range(self.n_iterations):
            # 미니배치 샘플 선택
            indices = np.random.choice(X.shape[0], size=self.batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            # 순방향 전파
            activations = self.forward_pass(X_batch, weights, bias)

            # 손실 계산
            loss = self.compute_loss(y_batch, activations[-1])

            # 역전파
            gradients = self.backward_pass(X_batch, y_batch, activations, weights, bias)

    def predict(self, X, weights, bias):
        activations = self.forward_pass(X, weights, bias)
        return activations[-1]

# 데이터 생성 및 분할
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 10%의 데이터에 결측값 추가
X.ravel()[np.random.choice(X.size, int(X.size * 0.1), replace=False)] = np.nan
y.ravel()[np.random.choice(y.size, int(y.size * 0.1), replace=False)] = np.nan

# 결측값을 평균값으로 대체
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 각 은닉층 수에 대해 모델 학습 및 평가
hidden_layer_sizes_list = [(10,), (10, 20), (10, 20, 30)]

for hidden_layer_sizes in hidden_layer_sizes_list:
    model = NeuralNetwork(hidden_layer_sizes=hidden_layer_sizes, learning_rate=0.01, n_iterations=1000, batch_size=32)
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test, model.weights, model.bias)
    mse = mean_squared_error(y_test, y_pred)

    # 결과 출력
    print(f"Hidden Layer Sizes: {hidden_layer_sizes}, MSE: {mse}")

    # 결과 시각화
    plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
    plt.plot(X_test, y_pred, label=f'Hidden Layer Sizes: {hidden_layer_sizes}', alpha=0.7)
    plt.title('Neural Network Regression with Varying Hidden Layer Sizes')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.show()
```
```python
Hidden Layer Sizes: (10,), MSE: 32.41469324515866
Hidden Layer Sizes: (10, 20), MSE: 32.0595348591773
Hidden Layer Sizes: (10, 20, 30), MSE: 63.97608226871222
```
![hidden layer result](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(8).png)

### 9. **합성곱 신경망(CNN)에서의 커널 크기와 스트라이드**:이미지 분류와 같은 작업에서 사용되는 CNN에서는 커널 크기와 스트라이드를 조절하여 특징을 추출하는 방식을 결정합니다.

>**CNN**: CNN은 영상 및 시계열 데이터에서 특징을 찾아내고 학습하기 위한 최적의 아키텍처를 제공한다. 이미지의 특징을 추출하는 부분과 클래스를 분류하는 부분으로 구성돼 있다. 특징을 추출하는 부분은 Convolutional Layer과 Pooling Layer를 겹겹이 쌓은 형태이다. CNN의 마지막 부분에는 이미지 분류(classification)을 위한 Fully Connected Layer이 추가된다.  https://rubber-tree.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-CNN-Convolutional-Neural-Network-%EC%84%A4%EB%AA%85
>
>![CNN 이미지](https://github.com/leejoohyunn/images/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202023-12-17%20212345.png)

>**스트라이드**: 입력 데이터에 커널(필터)을 적용할 때 이동할 간격을 조절하는 것이다. 다시 말해 커널(필터)가 이동할 간격이다. 스트라이드는 출력 데이터의 크기를 조절하기 위해 사용된다.
>
>![스트라이드 이미지](https://github.com/leejoohyunn/images/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202023-12-17%20211547.png) https://excelsior-cjh.tistory.com/79
>

(https://doug.tistory.com/44)

## 1.3 튜닝 방법 소개
### 1. **Manual Search**: 
rule of thumb 이라고도 하며, 경험 혹은 감으로 하이퍼파라미터 값을 조정하는 방법이다. 하이퍼파라미터별 흔히 알려져있는 값들을 사용하므로 편하지만, 하이퍼파라미터 조합별 성능을 비교하기 어여운 단점이 있다. 
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 나머지 파라미터 고정 및 시도할 학습률 값들
alpha = 0.0001  # 학습률을 수동으로 조절
learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]

# 각 학습률에 대해 모델 학습 및 평가
for lr in learning_rates:
    model = SGDRegressor(learning_rate='constant', eta0=alpha, max_iter=1000, random_state=42, alpha=0.0001, tol=1e-3, penalty=None)
    model.set_params(eta0=lr)  # 수동으로 학습률 변경
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 결과 출력
    print(f"Learning Rate: {lr}, MSE: {mse}")

# 최적 학습률을 사용하여 최종 모델 학습
best_lr = 0.01  # 최적의 학습률을 선택
final_model = SGDRegressor(learning_rate='constant', eta0=alpha, max_iter=1000, random_state=42, alpha=0.0001, tol=1e-3, penalty=None)
final_model.set_params(eta0=best_lr)
final_model.fit(X_train, y_train)

# 최종 모델의 예측 결과 시각화
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.plot(X_test, final_model.predict(X_test), label=f'Best Learning Rate: {best_lr}', alpha=0.7)
plt.title('Linear Regression with Manual Search for Learning Rate')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

```
arning Rate: 0.01, MSE: 104.97805743190273
Learning Rate: 0.05, MSE: 115.96412461861462
Learning Rate: 0.1, MSE: 112.82471835353513
Learning Rate: 0.5, MSE: 161.63202677218277
Learning Rate: 1.0, MSE: 429.13301012288537
```
![manual search](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(9).png)

### 2. **Grid Search**:
가능한 모든 조합의 하이퍼파라미터로 훈련시켜 최적의 조합을 찾는 방법이다. (exhaustive searching) 일부 파라미터는 범위가 없기 때문에 사용자가 경계를 지정해주기도 한다. 모든 가능성을 살펴보기 때문에 시간이 올래 걸린다는 단점이 있다.
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 파라미터 그리드 정의
param_grid = {
    'eta0': [0.0001, 0.001, 0.01, 0.1, 0.5],  # 시도할 학습률 값들
    'alpha': [0.0001],  # 나머지 파라미터 고정
    'max_iter': [1000],  # 나머지 파라미터 고정
    'penalty': [None],  # 나머지 파라미터 고정
}

# SGDRegressor 모델 생성
sgd = SGDRegressor(learning_rate='constant', random_state=42, tol=1e-3)

# GridSearchCV를 사용하여 최적의 학습률 찾기
grid_search = GridSearchCV(sgd, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# 최적의 학습률과 그때의 모델 출력
best_eta0 = grid_search.best_params_['eta0']
best_model = grid_search.best_estimator_
print(f"Best Learning Rate: {best_eta0}")

# 최종 모델의 예측 결과 시각화
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.plot(X_test, best_model.predict(X_test), label=f'Best Learning Rate: {best_eta0}', alpha=0.7)
plt.title('Linear Regression with Grid Search for Learning Rate')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```
```python
Best Learning Rate: 0.01
```
![grid result](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(10).png)

### 3. **Random Search**:
경계 내에서 임의의 조합을 추출해 최적의 조합을 찾는 방법이다. Grid Search에 비해 시간이 단축된다는 장점이 있다. 하지만 Grid Search와 마찬가지로 최적의 하이퍼파라미터를 위해 광범위한 범위를 탐색하기 때문에 비효율적이다. 
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 파라미터 분포 정의
param_dist = {
    'eta0': np.logspace(-5, 0, num=1000),  # 학습률을 로그 스케일로 정의
    'alpha': [0.0001],  # 나머지 파라미터 고정
    'max_iter': [1000],  # 나머지 파라미터 고정
    'penalty': [None],  # 나머지 파라미터 고정
}

# SGDRegressor 모델 생성
sgd = SGDRegressor(learning_rate='constant', random_state=42, tol=1e-3)

# RandomizedSearchCV를 사용하여 최적의 학습률 찾기
random_search = RandomizedSearchCV(sgd, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42)
random_search.fit(X_train, y_train)

# 최적의 학습률과 그때의 모델 출력
best_eta0 = random_search.best_params_['eta0']
best_model = random_search.best_estimator_
print(f"Best Learning Rate: {best_eta0}")

# 최종 모델의 예측 결과 시각화
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.plot(X_test, best_model.predict(X_test), label=f'Best Learning Rate: {best_eta0}', alpha=0.7)
plt.title('Linear Regression with Random Search for Learning Rate')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

```
```python
Best Learning Rate: 0.031878912926776456
```
![random result](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(11).png)

### 4. **Bayesian Optimization**:
목적함수를 최소 혹은 최대일 때 최적의 해를 구하는 방법이다. loss와 accuracy가 지표가 되어 우리가 구하고자하는 목적함수이다. 베이지안 최적화는 목적함수와 하이퍼파라미터로 Surrogate Model을 제작 및 평가하고 Acquisition Function으로 인풋 조합을 추천하는 과정을 반복하게 된다. 
![베이지안 최적화 이미지](https://github.com/leejoohyunn/images/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202023-12-17%20221133.png)

>surrogate model: 목적함수에 대한 확률적 추정 모델이다
>Acquisition Function: Surrogate model의 결과를 바탕으로 하이퍼파라미터 조합을 추천하는 함수이다.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 파라미터 분포 정의
param_dist = {
    'eta0': (1e-5, 1.0, 'log-uniform'),
    'alpha': (1e-5, 1e-1, 'log-uniform'),
    'max_iter': (100, 2000),  # 수정: 더 넓은 범위로 설정
    'penalty': [None],
}

# SGDRegressor 모델 생성
sgd = SGDRegressor(learning_rate='constant', random_state=42, tol=1e-3)

# BayesSearchCV를 사용하여 최적의 학습률 찾기
bayes_search = BayesSearchCV(sgd, search_spaces=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42)
bayes_search.fit(X_train, y_train)

# 최적의 학습률과 그때의 모델 출력
best_eta0 = bayes_search.best_params_['eta0']
best_model = bayes_search.best_estimator_
print(f"Best Learning Rate: {best_eta0}")

# 최종 모델의 예측 결과 시각화
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='black', marker='o', label='Test Data')
plt.plot(X_test, best_model.predict(X_test), label=f'Best Learning Rate: {best_eta0}', alpha=0.7)
plt.title('Linear Regression with Bayesian Optimization for Learning Rate')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```
```python
Best Learning Rate: 0.043513970791520494
```
![bayseian result](https://github.com/leejoohyunn/images/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20(12).png)

### 5. **Gradient-based Optimization**:
예측값과 실제값간 차이인 손실함수를 최소화하는 파라미터 조합을 찾는 방법이다. 이를 위해서는 가중치(weight)이나  편향(bias)를 업데이트해야합니다. 하지만 경사하강법의 한계로는 첫 번째, local minimum에 빠지기 쉽다는 것입니다. 두 번째, 안장점(Saddle point)를 벗어나지 못한다는 것이다. 

>최적의 가중치를 찾는 방법은 다음과 같다
>![가중치 구하는 이미지](https://github.com/leejoohyunn/images/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202023-12-17%20221820.png)


