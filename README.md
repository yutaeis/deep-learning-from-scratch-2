# deep-learning-from-scratch-2


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
