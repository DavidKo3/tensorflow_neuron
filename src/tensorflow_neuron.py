## refer goodtogreate.tistory.com/
import tensorflow as tf
import numpy as np

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w,x , name='output')
y_= tf.constant(0.0 , name ='correct_value') 
loss = tf.pow(y-y_, 2, name='loss')

train_step  = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for value in [x, w, y, y_,loss]:
    tf.scalar_summary(value.op.name, value)
    
summaries = tf.merge_all_summaries()

sess = tf.Session()
summary_writer = tf.train.SummaryWriter('log_simple_stats', sess.graph)
sess.run(tf.initialize_all_variables())

for i in range(150):
    if i % 10 ==0:
        print("epch {}, output : {} ".format(i, sess.run(y)))
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)
