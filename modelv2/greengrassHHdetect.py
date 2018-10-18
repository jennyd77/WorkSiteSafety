#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
""" A sample lambda for worksite safety detection"""
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk
import mo

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

def greengrass_infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        model_type = 'classification'
        
        ############################################################
        ###                                                      ###
        ###      Map inference return codes to outputs           ###
        ###      Map inference return codes to text colour       ###        
        ###                                                      ###
        ############################################################
        output_map = {0: 'compliant', 1: 'not compliant', 2: 'unknown', 3: 'no subject'}
        green=(0, 255, 0)
        red=(0, 0, 255)
        blue=(255, 0, 0)
        colour_map = {0: green, 1: red, 2: blue, 3: blue}
        # Create an IoT client for sending to messages to the cloud.
        client = greengrasssdk.client('iot-data')
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        
        ############################################################
        ###                                                      ###
        ###      Optimize model for Intel using clDNN            ###
        ###                                                      ###
        ############################################################
        model_name = 'image-classification'
        error, model_path = mo.optimize(model_name, 224, 224, aux_inputs={"--epoch":2})

        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading Worksite safety model to GPU')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Worksite safety model loaded')
        # Since this is a four class classifier only retrieve 4 classes.
        num_top_k = 4
        # The height and width of the training set images
        input_height = 224
        input_width = 224
        counter=0
        # Do inference until the lambda is killed.
        while True:
            ############################################################
            ###                                                      ###
            ###        Get a frame from the video stream             ###
            ###                                                      ###
            ############################################################
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            ############################################################
            ###                                                      ###
            ###    Resize frame to the same size as model expects    ###
            ###                                                      ###
            ############################################################
            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a classification model,
            # a simple API is provided.
            ############################################################
            ###                                                      ###
            ###           Perform inference against model            ###
            ###                                                      ###
            ############################################################
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            #client.publish(topic=iot_topic, payload="About to publish parsed_inference_results")
            ############################################################
            ###                                                      ###
            ###     Get top results with highest probabilities       ###
            ###                                                      ###
            ############################################################
            top_k = parsed_inference_results[model_type][0:num_top_k]
            print top_k
            adjusted_results = top_k
            
            
            # Add the label of the top result to the frame used by local display.
            # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
            # for more information about the cv2.putText method.
            # Method signature: image, text, origin, font face, font scale, colour (in BGR), and thickness
            ############################################################
            ###                                                      ###
            ### Set output text and text colour based on top result  ###
            ###                                                      ###
            ############################################################
            output=output_map[top_k[0]['label']]
            confidence=top_k[0]['prob']
            colour=colour_map[top_k[0]['label']]

            ############################################################
            ###                                                      ###
            ### If confidence is low, force result to unsure         ###
            ###                                                      ###
            ############################################################
            if ((output == "compliant" or output == "not compliant") and (confidence < 0.5)):
                adjusted_results[0]['label']=2
                adjusted_results[0]['prob']=1
                output=output_map[adjusted_results[0]['label']]
                colour=colour_map[adjusted_results[0]['label']]
                
            ############################################################
            ###                                                      ###
            ###        Superimpose text on the stored frame          ###
            ###                                                      ###
            ############################################################
            cv2.putText(frame, output, (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, colour, 8)
                        
            ############################################################
            ###                                                      ###
            ###      Display marked up frame in project stream       ###
            ###                                                      ###
            ############################################################
            local_display.set_frame_data(frame)
            # Every 10th inference, send the top k results to the IoT console via MQTT
            counter=counter+1
            if counter >= 10:
                cloud_output = {}
                for obj in top_k:
                    cloud_output[output_map[obj['label']]] = obj['prob']
                #client.publish(topic=iot_topic, payload="About to publish cloud_output")
                client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
                counter=0
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in worksite safety lambda: {}'.format(ex))

greengrass_infinite_infer_run()
