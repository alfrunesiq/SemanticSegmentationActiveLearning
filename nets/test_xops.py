import numpy as np
import tensorflow as tf

import extra_ops as xops

def test_unpool2d(sess):
    # Generate random images, and do:
    # max_pool -> max_unpool -> max_pool
    # assert first and last image to be equal
    imges  = np.random.uniform(size=[4,512,512,3])
    inputs = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32)

    mp, am = tf.nn.max_pool_with_argmax(inputs, ksize=[1,2,2,1], \
                                        strides=[1,2,2,1], padding="SAME")
    up = xops.unpool_2d(mp, am)
    mpupmp = tf.nn.max_pool(up, ksize=[1,2,2,1], \
                            strides=[1,2,2,1], padding="SAME")
    [out1, out2] = sess.run([mp, mpupmp], feed_dict={inputs: imges})
    err = np.sum(np.abs(out1 - out2))
    assert err == 0.0, "ERROR [test_unpool2d]: err=%f" % err

def main():
    with tf.Session() as sess:
        test_unpool2d(sess)

if __name__ == "__main__":
    main()
