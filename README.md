# deep-learning-from-scratch-2

### 20231126
2.4.3 SVDによる次元削減
```Python
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print(C[0]) #共起行列
print(W[0]) #MMPI matrix
print(U[0]) #SVD
```

### 20231013
- Sigmoid, Affine layer実装

```Python
#Sigmoid
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx 

#Affine
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out
    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0) 
        return dx       

```

### 20230930
- github repository 作成
- ch01: notebook作成
- ch01: 1.3.4 Repeat node, Sum node, MatMul

```Python
#Matmul
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW #3点リーダー... -> deep copy．メモリ位置を固定して配列要素を上書きする -> メモリ位置固定することでインスタンス変数gradsの扱いがシンプルになる
        return dx
```
