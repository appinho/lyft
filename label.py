import numpy as np
from PIL import Image
from scipy.ndimage.measurements import label
import os

# Define paths
path = './data/Train/'
file = 500
rgb_path = path + 'CameraRGB/' + str(file) + '.png'
label_path = path + 'CameraSeg/'

# Loop through training images
for filename in os.listdir(label_path):
	file = label_path + filename
	print(file)

	# Read images
	#image = Image.open(rgb_path)
	labeled_image = Image.open(file)

	# Binarize labeled image
	labeled_image = np.array(labeled_image)
	red_labeled_image = labeled_image[:,:,0]
	vehicle_out = np.where(red_labeled_image==10, 255, 0)
	road_out = np.where((red_labeled_image==6) | (red_labeled_image==7), 255, 0)

	# Remove front part of ego car
	structure = np.ones((3, 3), dtype=np.int)
	labeled, ncomponents = label(vehicle_out, structure)
	post_vehicle_out = np.where(labeled==ncomponents, 0, vehicle_out)

	# Merge both images
	red_channel = np.where(post_vehicle_out==255, 0, 255)
	labeled_image[:,:,0] = red_channel
	labeled_image[:,:,2] = road_out
	final_image = Image.fromarray(np.uint8(labeled_image))

	# Save images
	final_label_path = path + 'LabeledSeg/' + filename
	final_image.save(final_label_path)
