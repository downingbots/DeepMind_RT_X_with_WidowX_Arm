import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from rlds import transformations
from rlds import rlds_types
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
from IPython import display
from PIL import Image
import numpy as np
import math
import time
import datetime
import json
import camera_snapshot
import widowx_client
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from typing import Any, Dict, Union, NamedTuple
# from typing import Dict, Optional
# import transformation_utils as tr
# import reverb
# import tree
# import abc
# import dataclasses
# @title Transformation definitions

#########################################################################
# Functions extracted from DeepMind sample code:
# robotics_open_x_embodiment_and_rt_x_oss_Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb
#########################################################################
def as_gif(images, rbt=False):
  # Render the images as the gif:
  if rbt:
    filenm = '/tmp/temprbt.gif'
  else:
    filenm = '/tmp/temp.gif'

  images[0].save(filenm, save_all=True, append_images=images[1:], duration=1000, loop=0)
  gif_bytes = open(filenm,'rb').read()
  return gif_bytes

def resize(image):
    image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
    image = tf.cast(image, tf.uint8)
    return image

###################################################3
def read_config():
    with open('rt1_widowx_config.json') as config_file:
      config_json = config_file.read()
    config = json.loads(config_json)
    return config

def get_initial_state(config):
    init_state = json.loads(config["initial_state"])
    print("init_state", init_state)
    return init_state


###################################################3
# Start code for dataset trajectory example derived from:
# robotics_open_x_embodiment_and_rt_x_oss_Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb
# The original code uses the dataset to get one episode of images
# and state to get "gt - ground truth".  These images and state are
# then run with the model to get "predicted actions". No real robot
# arm was required.
#
# For this application, we run predicted actions, get images and states 
# using a real robot.
###################################################3
# some globals to make some of the below code easier to understand
x = 0
y = 1
z = 2
config = read_config()
saved_model_path = config["saved_model_path"]
run_tf = True
#######################################

# Load TF model checkpoint
if run_tf:
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
      tf.config.set_visible_devices(gpus[0], 'GPU')
      logical_gpus = tf.config.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
      # Visible devices must be set before GPUs have been initialized
      print(e)
else:
  tf.config.experimental.set_memory_growth
  print("memory growth set")

##########################################
# Perform one step of inference using dummy input
##########################################
tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
      model_path=saved_model_path,
      load_specs_from_pbtxt=True,
      use_tf_function=True)

# Obtain a dummy observation, where the features are all 0
observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation))

# Construct a tf_agents time_step from the dummy observation
tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))

# Initialize the state of the policy
policy_state = tfa_policy.get_initial_state(batch_size=1)

# Run inference using the policy
action = tfa_policy.action(tfa_time_step, policy_state)

# Create a dataset object to obtain episode from
# bridge_ds_name = config["bridge_ds_name"]
# bridge_dir = config["bridge_ds_dir"]

#######################################
# Move to initial positions and then take snapshot image
#######################################
robot_camera = None
robot_images = []
robot_arm =  widowx_client.WidowX()

robot_arm.move_to_position("To Point")
robot_camera = camera_snapshot.CameraSnapshot()

#########################################################
# Move to Initial Arm Position as taken from config file
#########################################################

state = get_initial_state(config)
print("initial config state:", state)
# [-127,127] for vx, vy and vz and [-255,255] for vg
# 41cm horizontal reach and 55cm verticle
# Values not normalized: already has the 127/255 factored in 
px = state["x"]
py = state["y"]
pz = state["z"]
pg = state["gamma"] 
pq5 = state["rot"] 
gripper_open = state["gripper"] 
[success,err_msg] = robot_arm.move(px, py, pz, pg, pq5, gripper_open)
robot_arm.move_to_position("By Point")
im, im_file, im_time = robot_camera.snapshot(True)
robot_image = Image.fromarray(np.array(im))
robot_images.append(im)

#########################################################
# Run as many steps as necessary
#########################################################
while True:
  #########################################################
  # set up input to Run Model to get predicted action
  #########################################################
  # BridgeData V2 dataset description:
  # step: <_VariantDataset element_spec= {
  # 'action': TensorSpec(shape=(7,), dtype=tf .float32, name=None),
  # 'discount': TensorSpec(shape=(), dtype=tf.float32, name=None),
  # 'is_first': TensorSpec(shape=(), dtype=tf.bool, name=None),
  # 'is_last': TensorSpec(shape=(), dtype=tf.bool, name=None),
  # 'is_terminal': TensorSpec(shape=(), dtype=tf.bool, name=None),
  # 'language_embedding': TensorSpec(shape=(512,), dtype= tf.float32, name=None),
  # 'language_instruction': TensorSpec(shape=(), dtype=tf.string, name=None),
  # 'observation':
  #    {'image_0': TensorSpec(shape=(256, 256, 3), dty pe=tf.uint8, name=None),
  #    'image_1': TensorSpec(shape=(256, 256, 3), dtype=tf.uin t8, name=None),
  #    'image_2': TensorSpec(shape=(256, 256, 3), dtype=tf.uint8, name= None),
  #    'image_3': TensorSpec(shape=(256, 256, 3), dtype=tf.uint8, name=None),
  #    'state': TensorSpec(shape=(7,), dtype=tf.float32, name=None)},
  # 'reward': TensorSpec(shape=(), dtype=tf.float32, name=None)}>
    
  predicted_actions = []
  print("instr:", config['language_instruction'])
  observation['natural_language_instruction'] = config['language_instruction']
  observation['image'] = robot_image

  # normalize observation state
  s = []
  s.append((state["x"]/127.0))
  s.append((state["y"]/127.0))
  s.append((state["z"]/127.0))
  s.append((state["gamma"]/255.0))
  s.append((state["rot"]/255.0))
  s.append(0)
  s.append(state["gripper"])
  # ts is timestep
  tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))
  
  #############################################################
  # Run inference using the policy rt_1_x
  #############################################################
  policy_step = tfa_policy.action(tfa_time_step, policy_state)
  
  ####################
  # result from rt_1_x
  robot_action = policy_step.action
  predicted_actions.append(robot_action)
  print("action:",robot_action)
  # ts is timestep class
  tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))
  
  policy_state = policy_step.state
  
  wv = robot_action['world_vector']
  euler = robot_action['rotation_delta']
  gripper_action = robot_action['gripper_closedness_action']
  if gripper_action < 0:
    gripper_open = True
  else:
    gripper_open = False
  print("wv:", wv)
  
  ############################################################
  # Move the robot based on predicted action and take snapshot
  ############################################################
  # denormalize action 
  # widowx has 41cm horizontal reach and 55cm verticle (up)
  vx = min(max(-127, round(wv[x] * 127.0 / 1.75)), 127)
  vy = min(max(-127, round(wv[y] * 127.0 / 1.75)), 127)
  vz = min(max(-127, round(wv[z] * 127.0 / 1.75)), 127)
  vg = min(max(-255, round(euler[0] * 255.0 / 1.4)), 255)
  vq5 = min(max(-255, round(euler[1] * 256.0 / 1.4)), 255)
  [success, err_msg] = robot_arm.move(vx, vy, vz, vg, vq5, gripper_open)
  if not success:
    print("Bad start point", px, py, pz, pg, pq5, gripper_open)
    print("err_msg:", err_msg)
    exit()
  
  im, im_file, im_time = robot_camera.snapshot(True)
  im = Image.fromarray(np.array(im))
  robot_images.append(im)
  
  observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation))
  tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))

  print('is_terminal:', robot_action['terminate_episode'])
  is_term = robot_action['terminate_episode']
  if is_term[0] == 1:
    break

display.Image(as_gif(robot_images, True))
print("sleeping")
time.sleep(180)

##################################
