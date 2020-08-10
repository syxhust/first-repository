import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(1, kernel_size=[2, 2], strides=[1, 1], padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        # 设计两层连接层的神经元个数，第一层无所谓，第二次一定和输出一致
        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense2 = tf.keras.layers.Dense(25)

    def call(self, inputs):
        z = self.conv1(inputs)
        z = self.flatten(z)
        z = self.dense1(z)
        z = self.dense2(z)
        # 返回原始5*5输出矩阵
        return tf.reshape(z, [-1, 5, 5])


# 使用Adam优化
optimizer = tf.optimizers.Adam()
epochs = 500
# 每批输入输入集的数量
batch_size = 50
model = Model()
correct_rate = []
for e in range(epochs):
    xs = []
    ys = []
    # 将每批的数据集中输入和标签整合在一起
    correct = 0
    # 每批训练1000个
    for b in range(batch_size):
        while 1:
            row, col = np.random.randint(5), np.random.randint(5)
            if (row == 1 | 2 | 3) & (col == 1 | 2 | 3):
                continue
            elif ((row == 4) & (col == 2)) | ((col == 4) & (row == 2)):
                continue
            else:
                break
        x = np.zeros([5, 5], 'float32')
        x[row, col] = 1
        xs.append(x)
        y = np.zeros([5, 5], 'float32')
        y[(row + 1) % 5, (col + 1) % 5] = 1
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)

    with tf.GradientTape() as t:
        xs = tf.reshape(xs, [50, 5, 5, 1])
        y_hat = model(xs)
        loss = tf.reduce_mean(tf.square(y_hat - ys))

    # 固定格式
    for b in range(batch_size):
        a = y_hat[b].numpy()
        c = ys[b]
        if (np.round(a) == np.round(c)).all():
            correct += 1
        else:
            continue
    gradients = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print('\r epoch:{}, loss:{}'.format(e, loss), '正确率=', correct / batch_size)
    correct_rate.append(correct / batch_size)

c = np.arange(1, 501)
plt.figure()
plt.plot(c, correct_rate)
plt.xlabel('epoch')
plt.ylabel('correct_rate')
plt.show()

credit = 0
error_first_first = 0
error_first_row_num = 0
error_first_col_num = 0
error_first_row_credit = 0
error_first_col_credit = 0
error_all = 0

for i in range(5):
    for j in range(5):
        test_x = np.zeros([1, 5, 5], 'float32')
        test_x[0, i, j] = 1
        test_x = tf.reshape(test_x, [1, 5, 5, 1])
        test_predict_true = model(test_x)
        test_predict = tf.round(model(test_x))
        test_y = np.zeros([1, 5, 5], 'float32')
        test_y[0, (i + 1) % 5, (j + 1) % 5] = 1
        if (np.round(test_y) == test_predict.numpy()).all():
            credit += 1
        else:
            error = sum(sum(sum(abs(test_y - test_predict_true.numpy()))))
            error_all += error
            print('输入为：', test_x.numpy(), '\n')
            print('训练出的为：', test_predict.numpy(), '\n')
            print('实际应该是：', test_y)
            print('\n')
            print('\n')
            if test_y[0, 0, 0] == 1:
                error_first_first = 1
            else:
                if sum(test_y[0, :, 0]) == 1:
                    error_first_col_num += 1
                    error = sum(sum(sum(abs(test_y - test_predict_true.numpy()))))
                    error_first_col_credit += error
                if sum(test_y[0, 0, :]) == 1:
                    error_first_row_num += 1
                    error = sum(sum(sum(abs(test_y - test_predict_true.numpy()))))
                    error_first_row_credit += error

print('得分为：', credit)
print('（4,4）位置错误个数为：', error_first_first)
print('输入中第五列错误个数为：', error_first_col_num)
print('输入中第五行错误个数为：', error_first_row_num)
print('输入中第五列错误的分值：', error_first_col_credit)
print('输入中第五行错误的分值：', error_first_row_credit)
print('总误差为：', error_all)

test_x = np.zeros([1, 5, 5], 'float32')
test_x[0, 1, 1] = 1
print(test_x)
test_x = tf.reshape(test_x, [1, 5, 5, 1])
test_predict = tf.round(model(test_x))
print(test_predict.numpy())