import tensorflow as tf
import os
import helper
import sys
import vgg

file = sys.argv[-1]

def run():

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:

        vgg_path = os.path.join(vgg.DATA_DIR, 'vgg')
        input, keep_prob, layer3, layer4, layer7 = vgg.load_vgg(sess, vgg_path)
        output = vgg.layers(layer3, layer4, layer7, vgg.NUM_CLASSES)
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, vgg.NUM_CLASSES))
        learning_rate = tf.placeholder(dtype = tf.float32)
        logits, train_op, cross_entropy_loss = vgg.optimize(output, correct_label, learning_rate, vgg.NUM_CLASSES)

    # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Restore variables from disk.
        saver.restore(sess, vgg.MODEL_DIR + "/cont_epoch_0.ckpt")
        print("Model restored.")

        if len(sys.argv) is 2:
            helper.save_inference_video_samples(
                file, vgg.RUNS_DIR, vgg.DATA_DIR, sess, vgg.IMAGE_SHAPE, logits, vgg.KEEP_PROB, input)
        else:
            helper.save_inference_samples(
                vgg.RUNS_DIR, vgg.DATA_DIR, sess, vgg.IMAGE_SHAPE, logits, vgg.KEEP_PROB, input)
if __name__ == '__main__':
    run()
