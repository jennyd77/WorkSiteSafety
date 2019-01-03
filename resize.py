# This program will take the images in your specified directory, 
# as well as the images in your subdirectory,
# and resize these images to your desired size.
# Where the aspect ratio is different in the original image to the desired size,
# the image will be cropped to suit.
# The height will be cropped from the bottom, and the width from both sides.

# Inputs:
# -i <Your directory>
# -x <The desired width>
# -y <The desired height>


#!/usr/local/bin/python3
from PIL import Image
import os
import cv2
import sys, getopt


def listdir_nohidden(path):
	for f in os.listdir(path):
		if not f.startswith('.'):
			yield f
# This function will rescale and crop the image to the desired size.
# If the height needs to be cropped, it will crop the bottom of the image away.
# If the width needs to be cropped, it will crop both sides away.
def resize_crop(im, desired_shp_x, desired_shp_y):
	y, x, channels = im.shape
	ratio_old = float(x) / float(y)
	ratio_new = float(desired_shp_x)/float(desired_shp_y)
	if ratio_old < ratio_new:
		newx, newy = desired_shp_x , float(y) * float(desired_shp_x) / float(x)
		newshape = (int(newx), int(newy))
		im2 = cv2.resize(im, newshape)
		# This will crop the function.
		im3 = im2[0:int(desired_shp_y), 0:int(desired_shp_x)]
	else:
		newx, newy = x * desired_shp_y / y, desired_shp_y
		newshape = (int(newx), int(newy))
		midpoint = newx/2.0
		im3 = im2.crop((midpoint - desired_shp_x/2.0, 0, midpoint + desired_shp_x/2.0, desired_shp_y))
	return im3

# The main function takes in the arguments for the image directory,
# and the desired size of the image.
def main(argv):
	try:
		opts, args = getopt.getopt(argv, "hi:x:y:", ["idir=", "desrd_x=", "desrd_y="])
	except getopt.GetoptError:
		print('resize.py -i <imagedir> -x <desrd_x> -y <dsrd_y>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('resize.py -x <desrd_x> -y <dsrd_y>')
			sys.exit()
		elif opt in ("-i", "--idir"):
			image_dir = arg
		elif opt in ("-x", "--desrd_x"):
			desired_x = arg
		elif opt in ("-y", "--desrd_y"):
			desired_y = arg

	# Here all subdirectories within the main directory are listed.
	# The files are then read in, resized with the resize_crop function,
	# and finally saved in a new directory called 'resized'.
	sub_dirs=listdir_nohidden(image_dir)
	for sub_dir in sub_dirs:
		if not sub_dir.endswith("jpg"):
			sub_dir_path=os.path.join(image_dir,sub_dir)
		else:
			sub_dir_path=image_dir
		files=listdir_nohidden(sub_dir_path)
		if files is not None:
			print('Processing the following directory:')
			print(sub_dir_path)
			for filename in files:
				# Reads files contained in directory.
				img = cv2.imread(os.path.join(sub_dir_path, filename))
				if img is not None:
					resized_img = resize_crop(img, desired_x, desired_y)
					# Creates directory called 'resized' if it doesn't already exist.
					new_path = os.path.join(sub_dir_path, 'resized')
					if not os.path.exists(new_path):
						os.makedirs(new_path)
					cv2.imwrite(os.path.join(new_path,filename), resized_img)




if __name__ == "__main__":
	main(sys.argv[1:])
