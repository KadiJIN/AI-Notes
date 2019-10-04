
#训练简单的线性模型


import numpy as np
import tensorflow as tf

sess = tf.Session()

#定义输入和期望的输出
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

#定义模型
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

#定义损失
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

#优化模型
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()
sess.run(init)

#迭代
for i in range(1000):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

#查看优化结果
print(sess.run(y_pred))  