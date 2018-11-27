Python 3.6.7 |Anaconda, Inc.| (default, Oct 28 2018, 19:44:12) [MSC v.1915 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> print ("hello")
hello
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, Tensoflow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
b'Hello, Tensoflow!'
>>> 
