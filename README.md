# WorkSiteSafety
Amazon Sagemaker notebook, image files, Lambda function for performing inference on DeepLens
The running model will identify the primary subject in view and determine whether they are compliant with WorkSite Safety standards for wearing a hardhat.

## Known Limitations
This model has only been trained with a clean, white hardhat. It will not currently produce reliable results for different coloured hardhats or for hardhats covered with stickers.<br/>
Currently I see this as a useful talking point about the need for training data that adequately represents the real-world use case.

The model has been trained at subject eye-level. A real-world deployment should consider the position of the camera (e.g. elevation) and ensure training images are captured at the correct angle

## Overview
This project is aimed to demonstrate the art of the possible for worksite safety.<br/>
Given what has been produced as a learning activity for one person with a couple of thousand images, imagine what is possible for a dedicated team!

## Algorithm
The algorithm chosen in this iteration of the project is the Amazon SageMaker built in Image Classification algorithm. This algorithm will look at an entire image and classify it as one of the following classifications:<br/>
* "Compliant" - The main subject in the image is wearing their hardhat and is hence compliant with that aspect of worksite safety.<br/>
* "Not compliant" - The main subject in the image is not wearing their hardhat and is hence not compliant with that aspect of worksite safety.<br/>
* "No Subject" - No subject was found in the image.<br/>
* "Unsure" - A subject is in the image but the algorithm is unable to determine compliance. The head may be cropped out of the image or obstucted.<br/>

A suggested future piece of work is to look at the new Amazon SageMaker built-in Object Detection algorithm. This would likely produce even better results for subjects standing a variable distance from the camera as well as for images with multiple subjects in view. The data preparation would however require a lot more effort.

## Training Data description
The training data used with this model was a set of images of an individual in near full-frame from their head down to around their thighs/knees. Some sample images have been provided in this repo; however, for privacy reasons no photos showing faces have been provided. Photos were taken on an iPhone with the camera held at eye-level. Subjects were asked to turn to each 45 degree point on the compass as well as look left and right at many of these compass points.<br/>
Photos were taken with the subject wearing the hard hat as well as a complimentary set of photos without the hardhat, with a baseball cap, or with the hardhat held in front of them or under their arm.

A real world scenario should pay careful attention to the location of the camera (will it be mounted overhead?) in order to gather images taken at the appropriate angle

## Creating training files
Photos were organized into the following directory structure

images<br/>
&nbsp;&nbsp;&nbsp;&nbsp;⎿	0\_compliant<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    ⎿ \<all compliant images><br/>
&nbsp;&nbsp;&nbsp;&nbsp; ⎿ 1\_notcompliant<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	 ⎿	\<all non-compliant images><br/>
&nbsp;&nbsp;&nbsp;&nbsp; ⎿ 2\_unsure<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	 ⎿	\<all images where head is cropped/obscured><br/>
&nbsp;&nbsp;&nbsp;&nbsp; ⎿ 3\_nosubject<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	 ⎿	\<all background/fuzz images><br/>

Next I used opencv to double my dataset by creating a mirror (vertical flip) of every image.<br/>
The python3 code used to perform this augmentation is provided as augment_images.py.<br/>

SageMaker's Image Classification algorithm has a preferred record type of recordio.

Download im2rec.py from [this github repo](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) or install mxnet (im2rec.py is included in the mxnet installation)

Run the following command to create .lst (list) and .idx (index) files for the dataset:<br/>
python im2rec.py --list --recursive --train-ratio 0.95 dataset images

Note: If you want to check how many images are in your dataset, use one of the following commands:<br/>
Linux/MAC: wc -l dataset\_train.lst<br>
Windows: findstr "." dataset\_train.lst | find /c/v ""<br>

Once complete, run the following command to create the recordio files which will be used as input for training our model:<br/>
python im2rec.py --resize 224 dataset images

## Upload training files to S3 for consumption by SageMaker
Create a S3 bucket in the us-east-1 (North Virginia). The reason is that DeepLens deployments are performed from this region so it is easiest to avoid having to transfer models between regions.

Copy the training and validation recordio files into separate directories in your S3 bucket:<br/>
aws s3 cp dataset\_train.rec s3://\<S3 bucket name>/train/<br/>
aws s3 cp dataset\_val.rec s3://\<S3 bucket name>/validation/<br/>

## Upload test files to S3 for testing inference in SageMaker notebook
It can be useful to have access to some test images in jpeg format that are already resized to the dimensions expected by your trained model. This will allow you to perform some preliminary testing of your trained model within your notebook before deploying the model to a DeepLens device

Store a handful of test images in a directory on your local machine (in this example the S3 directory is called test).<br/>
Copy the images to a directory in your S3 bucket:<br/>
aws s3 cp images\_test/ s3://\<S3 bucket name>/test/ --recursive

## Train your model
Using Amazon SageMaker in us-east-1 (North Virginia) region, create a notebook instance

Upload the notebook provided in this distribution "deeplens-worksite-safety-public.ipynb"

Find the line 'bucket_path="deeplens-your-bucketpath-name"' and change to match your bucket name and optionally a path in the bucket<br/>
**Note:** Your bucket name must start with the word 'deeplens' in order to allow the DeepLens IAM role to access your model.
Find the line 'num_layers = 50'. This is a hyperparameter you need to try different options for. You will need many images to take advantage of greater layers.<br/>
Find the line 'image_shape = "3,224,224"'. If you have changed the dimension of the images from 3 channels (RGB) with a maximum of 224x224, you will need to adjust this line<br/>
Find the line 'num_training_samples = 5308'. This must match the number of images in your training set<br/>
Find the line 'num_classes = 4'. This must match the number of output classifications<br/>
Find the line 'job_name_prefix = "your-jobname-prefix"'. Change this name to something relevant to your project<br/>

**Note:** mini_batch_size determines the number of images sent to each GPU at a time, ensure this setting is smaller than your training set.

Execute notebook cells down to but not including the section "Inference"<br/>

You now have a trained model!<br/>
You could now jump straight to deploying to your DeepLens device; however, the next step runs through some local tests using SageMaker to ensure the model is performing correctly

## Test model inference using SageMaker notebook
Find the line 'model_name="your-modelname"' and give it a unique model name

Execute notebook cells down to but not including the section "Download test images"<br/>

You now have an inference endpoint hosted by the SageMaker service!

Continue to the next cell in the notebook to download test images from the S3 bucket to this notebook local directory

In the section "Download test images" update the bucketpath to the location of your test images

In the section "Validate Model", find the line 'file\_name = "/tmp/test/sample\_image1.jpg"' and change the image name to match the image you wish to perform inference on

Execute the cell. Was the result correct? Was the probability nice and high?

## Delete the SageMaker hosted endpoint
Execute the final cell to delete (and stop paying for) the SageMaker hosting endpoint

# Deploy to your DeepLens
## Register your DeepLens device to your AWS account and ensure that the device status is Online
https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-getting-started-register.html


## Import your model
There are two options provided here:<br/>
1) Import your own model built using SageMaker<br/>
2) Import the sample pre-built model provided in this repository<br/>

### Option 1. Import your own model from SageMaker
From the DeepLens console, select 'Models'<br/>
Select 'Import Model'<br/>
From 'Import Source', ensure 'Amazon SageMaker trained model' is selected<br/>
From 'Model settings', select the following options:<br/>
* Amazon SageMaker training job ID: Choose the TRAINING JOB ID that produced your model (this will likely be the most recent training job)<br/>
* Model Name: Select a name that is meaningful to you <br/>
* Model Framework: MXNet

### Option 2. Import the sample pre-built model provided in this repository
From the DeepLens console, select 'Models'<br/>
Select 'Import Model'<br/>
From 'Import Source', select 'Externally Trained Model'<br/>
From 'Model settings', select the following options:<br/>
* Model artifact path: Select the S3 bucket path to the location of the model artifacts<br/>
&nbsp;&nbsp;&nbsp;&nbsp;Note1: The pathname must start with S3://deeplens)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;Note2: In this case the artifact will be model.tar.gz<br/>
* Model Name: Select a name that is meaningful to you<br/>
* Model Framework: MXNet

## Create the Lambda function which will run on the DeepLens device to perform inference
Follow the instructions provided at: https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-inference-lambda-create.html and create a function called "deeplens-hardhat-detection"<br/>
Note: ensure you copy the greengrass-hello-world blueprint as this includes libraries that are required

* Replace all code in the file "greengrassHelloWorld.py" with the code provided in the file "greengrassHHdetect.py"<br/>
* In your Lambda environment, change the name of your python function from "greengrassHelloWorld.py" to "greengrassHHdetect.py"<br/>
* Ensure that the Lambda function handler is specified as "greengrassHHdetect.function_handler"<br/>

Save your function<br/>
Publish your function "Actions - Publish new version"

## Create a DeepLens project
From the DeepLens console, select 'Projects'<br/>
Select 'Create new project'<br/>
From 'Project type', select 'Create a new blank project'<br/>
Select 'Next'<br/>
Within 'Project Information', select a Project Name that is meaningful to you<br/>
From 'Project Content', select the following options:<br/>
* Add model: Select the model you imported<br/>
* Add function: Select the Lambda function (at the required version) you published<br/>
Select 'Create' to create the project

## Deploy the project to your DeepLens device
From the DeepLens console, select 'Projects'<br/>
Select the radio button next to your project<br/>
Select 'Deploy to device'<br/>
Select the radio button next to your desired DeepLens device<br/>
Select 'Review'<br/>
When ready to deploy, select 'Deploy'<br/>

# Viewing project output
I found it very helpful to purchase a micro-HDMI to HDMI cable so that I could directly display the DeepLens output on a screen. Note that a USB keyboard and mouse will also be necessary if you wish to do this.<br/>
## There are two options to view the DeepLens project output:
### Option 1: View project stream on your DeepLens Device
https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-viewing-device-output-on-device.html#deeplens-viewing-output-project-stream
### Option 2: View Your AWS DeepLens Project Output in a Browser
https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-viewing-device-output-in-browser.html

# Update October 2018
I have published an alternate version of the Lambda function and model in the subfolder 'modelv2'. <br>
I am keen to hear your feedback on whether this new release produces better results.
The differences from v1 are:<br>
* To avoid saturating the IOT endpoint, only every 10th inference is sent to the endpoint<br>
* Additional images were added to cover edge cases<br>
* Lambda function is set to infer 'unsure' if neither 'compliant' or 'not compliant' are below 50% confidence<br>
* Color\_Crop\_Transform setting was enabled during training in order to augment the image set<br>
