# 手書き数字0~9の認識
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

# MNISTの手書き文字データセットの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label=True)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size =10)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01 # 学習率

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) # 1エポックの定義

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # ミニバッチの作成
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #誤差逆伝播法で勾配を算出
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        # 記録
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('{}エポック 訓練誤差:{} テスト誤差:{}'.format(int(i // iter_per_epoch), train_acc, test_acc))
        print('')

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

