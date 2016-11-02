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
    
