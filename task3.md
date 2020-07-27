
# 基于机器学习的文本分类

## 词向量+机器学习模型


```python
import os
print(os.getcwd())
```

    /Users/chenyifan/Desktop/研二/github/nlp_learning



```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeClassifier
# from sklearn.model_selection import train_test_split
```

**什么是TF-IDF**
TF-IDF(term frequency-inverse document frequency)词频-逆向文件频率。在处理文本时，如何将文字转化为模型可以处理的向量呢？TF-IDF就是这个问题的解决方案之一。字词的重要性与其在文本中出现的频率成正比(TF)，与其在语料库中出现的频率成反比(IDF)。


```python
train = pd.read_csv('./train_set.csv', sep = '\t', nrows = 100000)
```

## 1）count+RidgeClassifier


```python
vectorizer = CountVectorizer(max_features = 3000) #特征最多3000个
train_cot = vectorizer.fit_transform(train['text'])#训练的文本
#将每个词转换为一个离散向量
y_cot = train['label']
x_train_cot, x_valid_cot, y_train_cot, y_valid_cot = train_test_split(train_cot, y_cot, test_size = 0.3)
clf = RidgeClassifier()
clf.fit(x_train_cot,y_train_cot)

val_pred = clf.predict(x_valid_cot)
f1_score(val_pred, y_valid_cot, average = 'macro')
#训练1w条，验证1w条
```




    0.6859269497745054



## 2）TF-IDF+RidgeClassifier


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 3000)#特征最多3000个
train_tf = tfidf.fit_transform(train['text'])
y_train = train['label']
x_train_set, x_valid_set, y_train_set, y_valid_set = train_test_split(train_tf, y_train, test_size = 0.3)

clf2 = RidgeClassifier()
clf2.fit(x_train_set, y_train_set) #训练分类器

val_pred2 = clf2.predict(x_valid_set) #预测验证集
f1_score(y_valid_set, val_pred2, average = 'macro')
```




    0.8812642962950764



## 3）TF-IDF+Logistic regression


```python
from sklearn.linear_model import LogisticRegression
clf3 = LogisticRegression()
clf3.fit(x_train_set, y_train_set)
val_pred3 = clf3.predict(x_valid_set)
f1_score(y_valid_set, val_pred3, average = 'macro')
```




    0.8949276203912515


