
# 基于机器学习的文本分类




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
train = pd.read_csv('./train_set.csv', sep = '\t')
```


```python
vectorizer = CountVectorizer(max_features = 3000) #特征最多3000个
train_test = vectorizer.fit_transform(train['text'])#训练的文本
#将每个词转换为一个离散向量
clf = RidgeClassifier()
clf.fit(train_test[:10000], train['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
f1_score(train['label'].values[10000:], val_pred, average = 'macro')
#训练1w条，验证1w条
```




    0.6603842201513477


