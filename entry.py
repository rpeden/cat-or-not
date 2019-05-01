import tensorflow as tf
import sys

member_number = int(sys.argv[1])
print("The Round 2 entry code for member {0} is:".format(member_number))

one = [member_number, int(member_number/5), int(member_number/100)]

two = [0.02, 0.05, 0.08]

a = tf.placeholder(tf.float32, shape=(3))

b = tf.placeholder(tf.float32, shape=(3))

result = tf.tensordot(a, b, 1)

with tf.Session() as sess:

   print(int(result.eval(feed_dict={a: one, b: two})))