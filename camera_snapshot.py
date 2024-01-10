
#! /usr/bin/env python3

import os
import re
import shutil
import psutil
import subprocess
import time
import cv2
import copy
import numpy as np
from threading import Lock, Semaphore
import hashlib
import logging
import json
from IPython import display
from PIL import Image


##############################################################################
# Create 
class CameraSnapshot():
    def __init__(self):
        self.parse_param()
        self.full_resource_path = "/dev/video" + str(self._video_stream_provider)
        self.cap = None
        success = self.setup_capture_device()
        if not success:
            return
        self._buffer = []
        self.MAX_REPEATS = 100  # camera errors after 10 repeated frames in a row
        self._num_repeats = 0
        self._is_first_status = True
        self.start_time = time.time()

    # Only Public Interface
    def snapshot(self, save_img=False):
        # running at full speed the camera allows
        # while not rospy.is_shutdown():
        rval = False
        while not rval:
            for i in range(5):
              rval, frame = self.cap.read()
            if not rval:
                print(f"The frame has not been captured. You could try reconnecting the camera resource {self.full_resource_path}.")
                time.sleep(3)
                if self._retry_on_fail:
                    print(f"Searching for the device {self.full_resource_path}...")
                    self.setup_capture_device(exit_on_error=False)
            else:
                reading = [frame, time.time()]
                # if True:
                    # while(len(self._buffer) > self._buffer_queue_size):
                        # self._buffer.pop(0)
                    # self._buffer.append(reading)
                img = reading[0]
                img_time = reading[1]
                break
        file_nm = None
        dc_img = self.process_image(copy.deepcopy(img))
        if save_img:
            t = time.time()
            file_nm = self.store_latest_im(dc_img, t)
        return dc_img, file_nm, img_time


    #####################################
    # Private Interfaces Below
    #####################################
    def read_config(self):
        with open('rt1_widowx_config.json') as config_file:
          config_json = config_file.read()
        config = json.loads(config_json)
        return config

    def parse_param(self):
        camera_config = self.read_config()
        # self._fps = camera_config["video_fps"]
        self._height = camera_config["video_height"]
        self._width = camera_config["video_width"]
        self._retry_on_fail = True
        # self._buffer_queue_size = camera_config["buffer_queue_size"]
        self._topic_name = camera_config["camera_name"]
        self._video_stream_provider = camera_config["video_stream_provider"]
        self._cam_usb = camera_config["cam_usb"]
        self._image_dir = camera_config["image_dir"]

    def reset_camera_usb(self):
        # only 1 camera supported
        reset_names = [self._cam_usb]

        if shutil.which('usbreset') is None:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            res = subprocess.call(f'gcc {current_dir}/usbreset.c -o /usr/local/bin/usbreset')
            if not res == 0:
                print(f'usbreset install exit code: {res}')
                raise ValueError('could not install usbreset !')
        res = subprocess.run(['lsusb'], stdout=subprocess.PIPE)
        lines = res.stdout.decode().split('\n')
        for line in lines:
            for name in reset_names:
                if name in line:
                    numbers = re.findall(r'(\d\d\d)', line)[:2]
                    print('resetting usb with lsusb line: {}'.format(line))
                    cmd = 'sudo usbreset /dev/bus/usb/{}/{}'.format(*numbers)
                    res = subprocess.call(cmd.split())
                    if not res == 0:
                        print(f'exit code: {res}')
                        raise ValueError('could not reset !')
    
    def setup_capture_device(self):
        self.reset_camera_usb()
        success = False
        if not os.path.exists(self.full_resource_path):
            print("Device '%s' does not exist.", self.full_resource_path)
            return success
        # print("Trying to open resource: '%s'", self.full_resource_path)
        self.cap = cv2.VideoCapture(self.full_resource_path)
        if not self.cap.isOpened():
            print(f"Error opening resource: {self.full_resource_path}")
            print("opencv VideoCapture can't open it.")
            print("The device '%s' is possibly in use. You could try reconnecting the camera.", self.full_resource_path)
        if self.cap.isOpened():
            print(f"Correctly opened resource {self.full_resource_path}.")
            success = True
        return success

    def process_image(self, img):
        width = self._width
        height = self._height
        top = bot = right = left = 0
        # dtype = "bgr8"
        dtype = "bgr8"
        flip = True
        info_name = None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Check for overcrop conditions
        assert bot + top <= img.shape[0], "Overcrop! bot + top crop >= image height!"
        assert right + left <= img.shape[1], "Overcrop! right + left crop >= image width!"

        # If bot or right is negative, set to value that crops the entire image
        bot = bot if bot > 0 else -(img.shape[0] + 10)
        right = right if right > 0 else -(img.shape[1] + 10)

        # Crop image
        img = img[top:-bot, left:-right]

        # Flip image if necessary
        if flip:
            img = img[::-1, ::-1]

        # Resize image if necessary
        if (height, width) != img.shape[:2]:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return img

    def store_latest_im(self, img, img_time):
        # is the check for REPEATS too much in the synchronous mode?
        # Leaving in for now
        current_hash = hashlib.sha256(img.tobytes()).hexdigest()
        if self._is_first_status:
            self._height, self._width = img.shape[:2]
            self._is_first_status = False
            # self._status_sem.release()  # synchronous, one camera
        elif self._last_hash == current_hash:
            if self._num_repeats < self.MAX_REPEATS:
                self._num_repeats += 1
            else:
                logging.getLogger('robot_logger').error(f'Too many repeated images.\
                    Check camera with topic {self._topic_name}!')
                self.signal_shutdown('Too many repeated images. Check camera!')
        else:
            self._num_repeats = 0
        self._last_hash = current_hash

        # destroy window of prev image
        for proc in psutil.process_iter():
            # if proc.name() == "display":
            if proc.name() == "eog":
                proc.kill()
        t1 = int((img_time * 100000) % 10000000000000)
        print(self._image_dir + '/test_image_t{}.png'.format(t1))
        file_nm = self._image_dir + '/test_image_t{}.png'.format(t1)
        im = Image.fromarray(np.array(img))
        im.show()
        im.save(file_nm)
        # cv2.imwrite(file_nm, np.array(img))
        return file_nm

# for testing:
# cams = CameraSnapshot()
# while True:
#   dc_img, img_nm, img_time = cams.snapshot(True)
#   im0 = Image.fromarray(np.array(dc_img))
#   im0.show()
#   # cv2.imshow("image", np.array(dc_img))
#   time.sleep(1)
