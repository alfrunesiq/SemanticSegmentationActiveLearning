import argparse

import tensorflow as tf

import models

def main(args):
    file_contents  = tf.io.read_file(args.input_filename)
    input_image    = tf.image.decode_image(file_contents)
    image_shape    = tf.shape(input_image)
    crop_height    = image_shape[1] // 2
    slice_top_left = tf.stack([(image_shape[0] - crop_height) // 2, 0, 0])
    image_shape    = tf.stack([crop_height, image_shape[1], image_shape[2]])
    image_croped   = tf.slice(input_image, slice_top_left, image_shape)
    image          = tf.image.convert_image_dtype(image_croped, tf.float32)
    image          = tf.expand_dims(image, axis=0)
    image          = tf.image.resize_bilinear(image, [512,1024])
    input_var      = tf.get_variable("Input", shape=[1,512,1024,3])
    assign_input   = tf.assign(input_var, image)
    
    net = models.ENet(19)
    output = net(input_var, training=False)
    p_class = tf.nn.softmax(output)
    class_mask = tf.one_hot([0], 19, 1.0, 0.0)
    loss = tf.reduce_sum(p_class*p_class*class_mask)
    optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    apply_grads = optimizer.minimize(loss, var_list=[input_var])

    fw = tf.summary.FileWriter(args.log_dir)
    ckpt = tf.train.Checkpoint(model=net)
    print("Restoring model")
    status = ckpt.restore(args.checkpoint)

    image_summary = tf.summary.image("Image", input_var)

    with tf.Session() as sess:
        status.run_restore_ops(sess)
        print("Initializing input")
        sess.run(input_var.initializer)
        sess.run(assign_input)
        image = sess.run(image_summary)
        fw.add_summary(image)
        try:
            i = 0
            while True:
                print(i)
                i += 1
                if i % args.log_interval == 0:
                    _, img_sum, _loss = sess.run([apply_grads, image_summary, loss])
                    print(_loss)
                    fw.add_summary(img_sum, i // args.log_interval)
                else:
                    sess.run(apply_grads)
                
        finally:
            print("closing")
            fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-image",
            type=str,
            dest="input_filename",
            required=True,
            help="Path to input image")
    parser.add_argument("-l", "--log-dir",
            type=str,
            dest="log_dir",
            required=True,
            help="Path to log directory.")
    parser.add_argument("-c", "--checkpoint",
            type=str,
            dest="checkpoint",
            required=True,
            help="Path to model checkpoint.")
    parser.add_argument("-lr", "--learning-rate",
            type=float,
            dest="learning_rate",
            default=1e-3,
            help="Learning rate.")
    parser.add_argument("-li", "--log-interval",
            type=int,
            dest="log_interval",
            default=25,
            help="Log interval.")

    args = parser.parse_args()
    print(args)
    main(args)
