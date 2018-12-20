#!/usr/local/bin/python3
import os
import cv2
import sys, getopt

# This script will work through all subdirectories in a folder
# Each subdirectory should contain images files (and only image files)
# For each image, the script will create a replica of the image that is mirrored (flipped around an imaginary vertical line in the middle of the image)
# The vertically flipped replica of each image will have the same file name but pre-pended by vf_

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def main(argv):
   try:
      opts, args = getopt.getopt(argv,"hi:",["idir="])
   except getopt.GetoptError:
      print('augment_images.py -i <imagedir>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('augment_images.py -i <imagedir>')
         sys.exit()
      elif opt in ("-i", "--idir"):
         image_dir = arg


   #print("image_dir={}".format(image_dir))
   sub_dirs=listdir_nohidden(image_dir)
   for sub_dir in sub_dirs:
       sub_dir_path=os.path.join(image_dir,sub_dir)
       files=listdir_nohidden(sub_dir_path)
       for filename in files:
           #print(os.path.join(sub_dir_path,filename))
           # Only process images that are not already a vertical flip
           if not filename.startswith("vf_"):
               vf_filename="vf_" + filename
               #print("Checking for prescence of: {}".format(os.path.join(sub_dir_path,vf_filename)))
               # Only create a vertical flip of an image if it does not already have a vertical flip equivalent
               if not os.path.isfile(os.path.join(sub_dir_path,vf_filename)):
                   print("Processing {}".format(os.path.join(sub_dir_path,filename)))
                   img=cv2.imread(os.path.join(sub_dir_path,filename))
                   vertical_img = cv2.flip( img, 1 )
                   ##print(os.path.join(sub_dir_path,vf_filename))
                   cv2.imwrite(os.path.join(sub_dir_path,vf_filename),vertical_img)


if __name__ == "__main__":
   main(sys.argv[1:])

