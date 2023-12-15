import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

x_data=np.linspace(-0.5,0.5,400)[:,np.newaxis]
noise=np.random.normal(0.,0.03,x_data.shape)
y_data=np.power(x_data,3)+noise

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

w1=tf.Variable(tf.random_normal([1,40]))
b1=tf.Variable(tf.zeros([1,40]))
wb1=tf.matmul(x,w1)+b1
wbs1=tf.nn.tanh(wb1)

w2=tf.Variable(tf.random_normal([40,1]))
b2=tf.Variable(tf.zeros([1,1]))
wb2=tf.matmul(wbs1,w2)+b2
result=tf.nn.tanh(wb2)



loss=tf.reduce_mean(tf.square(y-result))
train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        print(i)
        sess.run(train,feed_dict={x:x_data,y:y_data})
    result=sess.run(result,feed_dict={x:x_data})

plt.figure()
plt.scatter(x_data,y_data)
plt.plot(x_data,result,'r-',lw=4)
plt.show()