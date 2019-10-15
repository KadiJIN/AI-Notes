import tensorflow as tf


sess = tf.Session()

my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(my_int_variable))