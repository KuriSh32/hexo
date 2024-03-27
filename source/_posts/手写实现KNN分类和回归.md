---
title: 手写实现KNN分类和回归
date: 2024-03-15 17:38:58
categories:
    - 机器学习
tags:
    - KNN
cover:
    /img/cover2.jpg
top_img:
    /img/top_img.jpg   
---

# Numpy实现KNN分类

## 功能

- 功能1：实现闵氏距离（Minkowski Distance），增加参数p来选择不同的距离度量（包括曼哈顿距离、欧几里得距离和切比雪夫距离）

- 功能2：对特征X增加标准化操作（包括标准化和归一化两种方法）

- 功能3：在iris数据集上datasets.load_iris()，测试不同距离度量和有没有使用标准化操作的效果（3*3共九种结果），使用准确率accuracy作为评估指标

  

## 实现




```python
# 分别对每一列的数值进行标准化处理
import numpy as np

def standardize(data):
    for col in range(len(data[0])):
        data_col = []
        for i in range(len(data)):
            data_col.append(data[i][col])
        mean = np.mean(data_col, axis=0)
        std = np.std(data_col, axis=0)
        for i in range(len(data)):
            data[i][col] = (data[i][col] - mean) / std        
    return data
```


```python
# 分别对每一列的数值进行归一化处理
def normalize(data):
    for col in range(len(data[0])):
        mx = data[0][col]
        mn = data[0][col]
        for i in range(len(data)):
            mx = max(mx, data[i][col])
            mn = min(mn, data[i][col])
        for i in range(len(data)):
            data[i][col] = (data[i][col] - mn) / (mx - mn)
    return data
```


```python
# 实现KNN分类器
import numpy as np
class KNNClassifier:
    def __init__(self, k, p=2):
        self.k = k
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _distance(self, x1, x2):
        # 如果p == -1， 则使用切比雪夫距离
        if self.p == -1:
            return np.max(np.abs(x1 - x2))
        else:
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def predict(self, X_test):
        y_pred = []
        for test_point in X_test:
            distances = [self._distance(test_point, train_point) for train_point in self.X_train]
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            unique_labels, label_counts = np.unique(nearest_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(label_counts)]
            y_pred.append(majority_label)
        return np.array(y_pred)
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        # 计算准确率
        accuracy = np.mean(y_pred == y_test)
        return accuracy
```


```python
# 加载数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 导入iris数据集
iris = load_iris()

# 提取特征和标签
X = iris.data  # 特征
y = iris.target  # 标签

# 定义随机种子
random_state = 42

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 打印训练集和测试集的大小
print("训练集大小:", X_train.shape, y_train.shape)
print("测试集大小:", X_test.shape, y_test.shape)

```

    训练集大小: (120, 4) (120,)
    测试集大小: (30, 4) (30,)



```python
# 使用九种模式进行预测
from sklearn import metrics
import copy

# 定义k值
k = 5

# 定义不同的p值
p_values = [1, 2, -1]

# 定义不同的处理操作
preprocess_options = ['null', 'standardize', 'normalize']

# 在不同的p值下进行测试
for p in p_values:
    # 使用不同的处理操作
    for preprocess in preprocess_options:
        # 输出p的不同取值和预处理的不同操作
        print(f"p={p}, preprocess={preprocess}")
        
        X_train_cur = copy.deepcopy(X_train)
        X_test_cur = copy.deepcopy(X_test)
        # 对数据进行不同的预处理
        if preprocess == 'null':
            pass
        elif preprocess == 'standardize':
            X_train_cur = standardize(X_train_cur)
            X_test_cur = standardize(X_test_cur)
        elif preprocess == 'normalize':
            X_train_cur = normalize(X_train_cur)
            X_test_cur = normalize(X_test_cur)
        
        # 创建KNN分类器
        knn_classifier = KNNClassifier(k=k, p=p)
        # 在KNN上进行训练
        knn_classifier.fit(X_train_cur, y_train)
        # 预测结果
        y_train_pred = knn_classifier.predict(X_train_cur)
        y_test_pred = knn_classifier.predict(X_test_cur)
        
        # 计算损失和准确率
        train_err = metrics.mean_squared_error(y_train, y_train_pred)
        test_err = metrics.mean_squared_error(y_test, y_test_pred)
        print( 'The mean squar error of train and test are: {:.2f}, {:.2f}'.format(train_err, test_err))
        predict_score = knn_classifier.score(X_test_cur,y_test)
        print('The decision coefficient is: {:.2f}'.format(predict_score))
        
        print(end='\n')
    print(end='\n')
```
## 测试结果

```

    p=1, preprocess=null
    The mean squar error of train and test are: 0.03, 0.00
    The decision coefficient is: 1.00
    
    p=1, preprocess=standardize
    The mean squar error of train and test are: 0.04, 0.03
    The decision coefficient is: 0.97
    
    p=1, preprocess=normalize
    The mean squar error of train and test are: 0.06, 0.00
    The decision coefficient is: 1.00


​      

    p=2, preprocess=null
    The mean squar error of train and test are: 0.03, 0.00
    The decision coefficient is: 1.00
    
    p=2, preprocess=standardize
    The mean squar error of train and test are: 0.04, 0.03
    The decision coefficient is: 0.97
    
    p=2, preprocess=normalize
    The mean squar error of train and test are: 0.04, 0.00
    The decision coefficient is: 1.00


​    

    p=-1, preprocess=null
    The mean squar error of train and test are: 0.03, 0.00
    The decision coefficient is: 1.00
    
    p=-1, preprocess=standardize
    The mean squar error of train and test are: 0.04, 0.07
    The decision coefficient is: 0.93
    
    p=-1, preprocess=normalize
    The mean squar error of train and test are: 0.03, 0.00
    The decision coefficient is: 1.00

```



# Numpy实现KNN回归

## 功能

- 功能1：实现闵氏距离（Minkowski Distance），增加参数p来选择不同的距离度量（包括曼哈顿距离、欧几里得距离和切比雪夫距离）

- 功能2：对特征X增加标准化操作（包括标准化和归一化两种方法）

- 功能3：在diabetes数据集上datasets.load_diabetes()，测试不同距离度量和有没有使用标准化操作的效果（3*3共九种结果），使用均方根误差（RMSE，自己实现）作为评估函数

## 实现

```python
# 分别对每一列的数值进行标准化处理
import numpy as np

def standardize(data):
    for col in range(len(data[0])):
        data_col = []
        for i in range(len(data)):
            data_col.append(data[i][col])
        mean = np.mean(data_col, axis=0)
        std = np.std(data_col, axis=0)
        for i in range(len(data)):
            data[i][col] = (data[i][col] - mean) / std        
    return data
```


```python
# 分别对每一列的数值进行归一化处理
def normalize(data):
    for col in range(len(data[0])):
        mx = data[0][col]
        mn = data[0][col]
        for i in range(len(data)):
            mx = max(mx, data[i][col])
            mn = min(mn, data[i][col])
        for i in range(len(data)):
            data[i][col] = (data[i][col] - mn) / (mx - mn)
    return data
```


```python
# 实现KNN回归器
# 基本复用KNN分类器的代码，仅在predict部分和score部分稍作修改，并添加rmse方法
import numpy as np
import math

class KNNRegressor:
    def __init__(self, k, p=2):
        self.k = k
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _distance(self, x1, x2):
        # 如果p == -1， 则使用切比雪夫距离
        if self.p == -1:
            return np.max(np.abs(x1 - x2))
        else:
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def predict(self, X_test):
        y_pred = []
        for test_point in X_test:
            distances = [self._distance(test_point, train_point) for train_point in self.X_train]
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            # KNN 回归的预测值为 k 个最近邻居的平均值
            prediction = np.mean(nearest_labels)
            y_pred.append(prediction)
        return np.array(y_pred)
    
    # 使用python标准库库实现rmse
    def rmse(self, y_true, y_pred):
        sum = 0
        for i in range(len(y_true)):
            sum += (y_true[i] - y_pred[i])**2
        mean = sum / len(y_true)
        rmse = math.sqrt(mean)
        return rmse
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        # 计算rmse
        rmse = self.rmse(y_test, y_pred)
        return rmse
```


```python
# 加载数据集
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 导入diabetes数据集
diabetes = load_diabetes()

# 提取特征和标签
X = diabetes.data  # 特征
y = diabetes.target  # 标签

# 定义随机种子
random_state = 42

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 打印训练集和测试集的大小
print("训练集大小:", X_train.shape, y_train.shape)
print("测试集大小:", X_test.shape, y_test.shape)
```

    训练集大小: (353, 10) (353,)
    测试集大小: (89, 10) (89,)



```python
# 使用九种模式进行预测
from sklearn import metrics
import copy

# 定义k值
k = 5

# 定义不同的p值
p_values = [1, 2, -1]

# 定义不同的处理操作
preprocess_options = ['null', 'standardize', 'normalize']

# 在不同的p值下进行测试
for p in p_values:
    # 使用不同的处理操作
    for preprocess in preprocess_options:
        # 输出p的不同取值和预处理的不同操作
        print(f"p={p}, preprocess={preprocess}")
        
        X_train_cur = copy.deepcopy(X_train)
        X_test_cur = copy.deepcopy(X_test)
        # 对数据进行不同的预处理
        if preprocess == 'null':
            pass
        elif preprocess == 'standardize':
            X_train_cur = standardize(X_train_cur)
            X_test_cur = standardize(X_test_cur)
        elif preprocess == 'normalize':
            X_train_cur = normalize(X_train_cur)
            X_test_cur = normalize(X_test_cur)
        
        # 创建KNN回归器
        knn_regressor = KNNRegressor(k=k, p=p)
        # 在KNN上进行训练
        knn_regressor.fit(X_train_cur, y_train)
        # 预测结果
        y_train_pred = knn_regressor.predict(X_train_cur)
        y_test_pred = knn_regressor.predict(X_test_cur)
        
        # 计算损失和准确率
        train_err = metrics.mean_squared_error(y_train, y_train_pred)
        test_err = metrics.mean_squared_error(y_test, y_test_pred)
        print( 'The mean squar error of train and test are: {:.2f}, {:.2f}'.format(train_err, test_err))
        predict_score = knn_regressor.score(X_test_cur,y_test)
        print('The decision coefficient is: {:.2f}'.format(predict_score))
        
        print(end='\n')
    print(end='\n')
```
## 测试结果

```

    p=1, preprocess=null
    The mean squar error of train and test are: 2651.05, 2925.80
    The decision coefficient is: 54.09
    
    p=1, preprocess=standardize
    The mean squar error of train and test are: 2639.52, 3210.57
    The decision coefficient is: 56.66
    
    p=1, preprocess=normalize
    The mean squar error of train and test are: 2659.10, 3438.37
    The decision coefficient is: 58.64


​    

    p=2, preprocess=null
    The mean squar error of train and test are: 2528.59, 3019.08
    The decision coefficient is: 54.95
    
    p=2, preprocess=standardize
    The mean squar error of train and test are: 2553.65, 3111.98
    The decision coefficient is: 55.79
    
    p=2, preprocess=normalize
    The mean squar error of train and test are: 2547.91, 3275.45
    The decision coefficient is: 57.23


​    

    p=-1, preprocess=null
    The mean squar error of train and test are: 2591.17, 3134.99
    The decision coefficient is: 55.99
    
    p=-1, preprocess=standardize
    The mean squar error of train and test are: 2637.77, 2981.47
    The decision coefficient is: 54.60
    
    p=-1, preprocess=normalize
    The mean squar error of train and test are: 2582.24, 3289.01
    The decision coefficient is: 57.35


​    
​  