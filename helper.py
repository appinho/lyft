import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import skvideo.io

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths = glob(os.path.join(data_folder, 'LabeledSeg', '*.png'))
        back_color = np.array([255, 0, 0])
        road_color = np.array([255, 0, 255])
        cars_color = np.array([0, 0, 0])

        #random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for i,image_file in enumerate(image_paths[batch_i:batch_i+batch_size]):
                gt_image_file = label_paths[i]
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_back = np.all(gt_image == back_color, axis=2)
                gt_road = np.all(gt_image == road_color, axis =2)
                gt_cars = np.all(gt_image == cars_color, axis =2)
                gt_back = gt_back.reshape(*gt_back.shape, 1)
                gt_road = gt_road.reshape(*gt_road.shape, 1)
                gt_cars = gt_cars.reshape(*gt_cars.shape, 1)
                gt_image = np.concatenate((gt_back, gt_road, gt_cars), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    batch_size = 5
    num_pixels = image_shape[0] * image_shape[1]
    org_image_shape = (600, 800)
    image_files = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    print(len(image_files))
    n_loop = int(len(image_files) / batch_size)
    print(n_loop)
    for i in range(n_loop):
        print("Batch", i)
        start_idx = i * batch_size
        stop_idx = (i+1) * batch_size
        images = []
        image_names = []
        for image_file in image_files[start_idx:stop_idx]:
            print(image_file)
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            images.append(image)
            image_names.append(image_file)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: images})[0]

        for i in range(batch_size):
            print("Infer",i)
            start_idx = i * num_pixels
            stop_idx = (i+1) * num_pixels
            max_class = np.argmax(im_softmax[start_idx:stop_idx], axis=1).reshape(image_shape[0], image_shape[1])
            road = (max_class == 1).reshape(image_shape[0], image_shape[1], 1)
            cars = (max_class == 2).reshape(image_shape[0], image_shape[1], 1)
            road_mask = np.dot(road, np.array([[0, 255, 0, 127]]))
            cars_mask = np.dot(cars, np.array([[255, 0, 0, 127]]))
            road_mask = scipy.misc.toimage(road_mask, mode="RGBA")
            cars_mask = scipy.misc.toimage(cars_mask, mode="RGBA")
            street_im = scipy.misc.toimage(images[i])
            street_im.paste(road_mask, box=None, mask=road_mask)
            street_im.paste(cars_mask, box=None, mask=cars_mask)
            
            res_image = scipy.misc.imresize(street_im, org_image_shape)

            yield os.path.basename(image_names[i]), np.array(res_image)

def gen_test_video_output(file, sess, logits, keep_prob, image_pl, runs_dir, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    video = skvideo.io.vread(file)
    for i,image_file in enumerate(video):
        print(i)
        image = scipy.misc.imresize(image_file, image_shape)
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        scipy.misc.imsave("./runs/" + str(i) + ".png", image)

def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'Test'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def save_inference_video_samples(file, runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images')
    image_outputs = gen_test_video_output(
        file, sess, logits, keep_prob, input_image, runs_dir, image_shape)
