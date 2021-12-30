# レイヤ（層）を定義
import numpy as np
from functions import *

# ReLU関数のクラス
class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0) # 各要素はbool値になる
        out = x.copy()
        out[self.mask] = 0 # x <= 0である要素（True）は0になった
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# Sigmoid関数のクラス
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) *self.out
        return dx
    

# ネットワークの順伝播で行う行列の積（アフィン変換）の処理を定義
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape  = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b  #順伝播の出力
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0) # axis = 0方向の和をとる
        dx = dx.reshape(*self.original_x_shape) # 入力データの形状に戻す(テンソル対応)
        return dx

# Softmax層とLoss(cross-entrop-error)層
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # y=softmax(x)
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size

        return dx

