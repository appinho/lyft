import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


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
        kernel_initializer= tf.random_normal_initializer(stddev=kernel_std),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(kernel_w))
    c_1x1_l4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=kernel_std),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(kernel_w))
    c_1x1_l7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=kernel_std),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(kernel_w))

    # Upsampling
    up_l7 = tf.layers.conv2d_transpose(c_1x1_l7, num_classes, 4, strides=(2, 2), padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=kernel_std),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(kernel_w))

    # 1st Skip connection
    skip_1 = tf.add(c_1x1_l4, up_l7)

    # Upsamling
    up_s1 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, strides=(2, 2), padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=kernel_std),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(kernel_w))

    # 2nd Skip connection
    skip_2 = tf.add(c_1x1_l3, up_s1)

    # Upsamling
    final_layer = tf.layers.conv2d_transpose(skip_2, num_classes, 16, strides=(8, 8), padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=kernel_std),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(kernel_w))

    # Return
    return final_layer

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    # Reshape so each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = labels, logits = logits))

    # Define optimizer
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

    # Return
    return logits, train_op, cross_entropy_loss


def optimize2(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    # Reshape so each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # your class weights
    class_weights = tf.constant([[1.0, 1.0, 1.0]])

    # deduce weights for batch samples based on their true label
    weights = tf.reduce_sum(logits * class_weights, axis=1)

    # compute your (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    unweighted_losses = tf.reshape(unweighted_losses, (-1, 3))
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * class_weights

    # reduce the result to get your final loss
    cross_entropy_loss = tf.reduce_mean(weighted_losses)

    # Define optimizer
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

    # Return
    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Session starts with op that initialize global variables
    sess.run(tf.global_variables_initializer())

    # Define hyperparameters for training
    p_keep = 0.50
    l_rate = 0.0001

    print("Training started")
    # Loop over epochs
    for epoch in range(epochs):

        print("Epoch: ", epoch + 1)

        i = 0
        # Loop over batch size
        for image, label in get_batches_fn(batch_size):
                
            # Train
            _, loss = sess.run([train_op, cross_entropy_loss], 
                feed_dict={input_image: image, correct_label: label,
                    keep_prob: p_keep, learning_rate: l_rate})

            i += 1
            print("Loss ", i, " : ", loss)

    print("Training finished")

def run():
    num_classes = 3
    image_shape = (256, 384)
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Hyperparameters
    num_epochs = 20
    batch_size = 1

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Define tf placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes],
            name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32,
            name = 'learning_rate')

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess,vgg_path)
        final_layer = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize2(final_layer, correct_label, 
            learning_rate, num_classes)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'SingleTrain'), image_shape)
        
        # Train NN using the train_nn function
        train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
            input_image, correct_label, keep_prob, learning_rate)

        # Save the variables to disk.
        save_path = saver.save(sess, "./tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)


        # Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video




if __name__ == '__main__':
    run()