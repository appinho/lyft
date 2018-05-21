import sys
import skvideo.io
import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import os
import scipy.misc


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # Define names
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Loads the model VGG from a SavedModel
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Load the default graph for the current thread
    graph = tf.get_default_graph()
    t_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    t_keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    t_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    t_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    t_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # Return
    return t_input, t_keep, t_layer3, t_layer4, t_layer7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Define kernel parameters
    kernel_w = 1e-3
    kernel_std = 0.01

    # 1x1 convolutions
    c_1x1_l3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                kernel_initializer=tf.random_normal_initializer(
                                    stddev=kernel_std),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_w))
    c_1x1_l4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                kernel_initializer=tf.random_normal_initializer(
                                    stddev=kernel_std),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_w))
    c_1x1_l7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                kernel_initializer=tf.random_normal_initializer(
                                    stddev=kernel_std),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_w))

    # Upsampling
    up_l7 = tf.layers.conv2d_transpose(c_1x1_l7, num_classes, 4, strides=(2, 2), padding='same',
                                       kernel_initializer=tf.random_normal_initializer(
                                           stddev=kernel_std),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_w))

    # 1st Skip connection
    skip_1 = tf.add(c_1x1_l4, up_l7)

    # Upsamling
    up_s1 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, strides=(2, 2), padding='same',
                                       kernel_initializer=tf.random_normal_initializer(
                                           stddev=kernel_std),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_w))

    # 2nd Skip connection
    skip_2 = tf.add(c_1x1_l3, up_s1)

    # Upsamling
    final_layer = tf.layers.conv2d_transpose(skip_2, num_classes, 16, strides=(8, 8), padding='same',
                                             kernel_initializer=tf.random_normal_initializer(
                                                 stddev=kernel_std),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_w))

    # Return
    return final_layer


file = sys.argv[-1]

if file == 'demo.py':
    print("Error loading video")
    quit

# Define encoder function


def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

image_shape = (256, 384)
org_image_shape = (600, 800)
data_dir = './data'
num_classes = 3
with tf.Session() as sess:

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    # Build NN using load_vgg, layers, and optimize function
    input_image, keep_prob, layer3, layer4, layer7 = load_vgg(
        sess, vgg_path)

    final_layer = layers(layer3, layer4, layer7, num_classes)
    logits = tf.reshape(final_layer, (-1, num_classes))

    # Restore variables from disk.
    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/model.ckpt")

    for rgb_frame in video:

        # Resize
        image = scipy.misc.imresize(rgb_frame, image_shape)

        im_softmax = sess.run([tf.nn.softmax(logits)], {
                              keep_prob: 1.0, input_image:    [image]})[0]
        max_class = np.argmax(im_softmax, axis=1).reshape(
            image_shape[0], image_shape[1])
        #print(max_class)
        unique, counts = np.unique(max_class, return_counts=True)
        road = (max_class == 1).reshape(image_shape[0], image_shape[1], 1)
        cars = (max_class == 2).reshape(image_shape[0], image_shape[1], 1)

        road = road[:, :, 0]
        cars = cars[:, :, 0]
        res_road = scipy.misc.imresize(road, org_image_shape)
        res_cars = scipy.misc.imresize(cars, org_image_shape)
        #scipy.misc.imsave("./runs/image" + str(frame) + ".png", rgb_frame)
        answer_key[frame] = [encode(res_cars), encode(res_road)]

        # Increment frame
        frame += 1

# Print output in proper json format
#print(json.dumps(answer_key))
