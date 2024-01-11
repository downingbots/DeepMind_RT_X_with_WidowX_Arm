# USING DEEPMIND'S RT-X MODEL WITH A LOW-END ROBOT ARM

The purpose of the DeepMind dataset was to illustrate generalization of robot arm control by training on the largest robot dataset thus-far collected.  This repository provides a bare-bones implementation to run the RT-X Models with a low-end robot arm. The laptop code is python-only and doesn't use ROS.  Hopefully, this repository will lower the barrier of entry for others to experiment with the most advanced text-to-robot-action models currently available. This code fills in some of the gaps necessary to drive a robot arm from DeepMind's minimal example.  However, the experimental results were disappointing. I had hoped that the RT-X model would be able to better generalize to a "previous generation" variation of the BridgeData hardware configuration.  Oh well.

## BACKGROUND

I previously experimented with reinforcement learning using UC Berkeley's REPLAB [1] hardware and software, which used the Interbotix WidowX Mark II in an overhead-arm configuration. 

More recently, UC Berkeley released BridgeData V2 [2][6], a large and diverse dataset of robotic manipulation behaviors designed to facilitate research in scalable robot learning. This dataset used a more recent low-cost (and larger) Interbotix WidowX 250 arm and a revised hardware configuration. The REPLAB configuration was easily modified to closeley match the BridgeData hardware configuration.

The BridgeData V2 dataset was incorporated into Google DeepMind's Open X-Embodiment: Robotic Learning Datasets and RT-X Models [3][7]. DeepMind assembled a dataset from 22 different robots collected through a collaboration between 21 institutions, demonstrating 527 skills (160266 tasks). RT-X is a high-capacity model trained on this data and exhibits positive transfer and improves the capabilities of multiple robots by leveraging experience from other platforms. Pretrained RT-1-X and RT-2-X models were released.  RT-1-X is an efficient Transformer-based architecture designed for robotic control, and RT-2-X is a large vision-language model co-fine-tuned to output robot actions as natural language tokens.  

REPLAB datasets also appear to be incorporated into the Open X dataset (an image of it appears in the jupiter notebook "robotics_open_x_embodiment_and_rt_x_oss_Open_X_Embodiment_Datasets.ipynb#Combination-of-multiple-datasets".)[4]

In this repository, I reconfigured the REPLAB configuration to be similar to the BridgeData configuration with the WidowX Mark II.[5]  I use Lenin Silva Gutiérrez's WidowX Mark II Arbotix Controller [6], and modified it to match to what is required by the output RT-X.  Using software adapted from Berkeley and DeepMind, I then put together greatly simplified experimental software that runs the RT-X model to control the Widow X Mark II using a realsense camera.

The RT-1-X and RT-2-X models output robot actions represented with respect to the robot gripper frame. The robot action is a 7-dimensional vector consisting of x, y, z, roll, pitch, yaw, and gripper opening or the rates of these quantities. These vectors are then used to control the WidowX Mark II.

According to the DeepMind paper [8], RT-1-X outperforms the original BridgeData V2 model with a large average improvement, demonstrating domains with limited data benefit substantially from co-training on X-embodiment data.  My results were underwhelming.

## HOW TO USE:

All the required source files are located in this single directory repository, but really there are two compute environments: (1) the System76 Laptop with Nvidia board and (2) the interbotix board on the WidowX Mark II Arm.
  
First, download DeepMind's RT-X model.  Follow the instructions to run the Minimal Example.

To run on the laptop, just execute "python rt1_widowx.py":
 - rt1_widowx.py: The laptop-side code (just execute "python rt1_widowx.py".) Largely derived from DeepMind sample code, it does the following:
   1. Sets up the RT-X model
   2. Moves the arm into an initial pose
   3. performs a closed loop to capture an image, feed the image into RT-X, executes the robot arm action determined by RT-X.  
 - rt1_widowx_config.json : a simple JSON config file. Change the "language_instruction" to change the test on a real robot arm.
 - widowx_client.py : client code to communicate with the interbotix board. Largely derived from LeninSG21/WidowX github code. 
 - camera_snapshot.py : a non-ROS version to capture a snapshot from a single realsense camera. Largely derived from the Berkeley Bridge Dataset V2 code.

For the arduino IDE and the interbotix board, follow the WidowX Mark II README by Lenin Silva Gutiérrez to install or understand much of the software, but include the following customized files instead: 
 - WidowX.cpp : modified to goto initial starting point and return the state of the arm.  Note, the initial position of the robot arm is hard-coded as the IK code for the initial arm pose from LeninSG21/WidowX didn't work for me except for cases when the gamma specified a horizontal gripper.
 - WidowX.h : trivially modified.
 - MoveWithController.ino : modified to goto initial starting position, and return required state.

## RESULTS

The results are disappointing.  The RT-X model appears to be properly controlling the robot arm and seems to perform Robot arm actions inspired by the "language instruction" and the image. Unfortunately, it appears to be pure luck if it actually performs a successful action.  Potential reasons for this:
  - The robot arm and its configuration is not an exact match to that used to gather much of the BridgeData dataset.  Some RT-X data may have been gathered using the WidowX Mark II robot arm, but using an overhead configuration. Most of the BridgeData dataset used a next-generation and longer low-end robot arm.  
  - The test objects and language instructions are based upon those used to gather the BridgeData dataset.  However, they were close but not exact matches.
  - Bugs in my code are still being found.

Yet, the purpose of the DeepMind dataset was to illustrate generalization of robot arm control by being trained on the largest robot dataset thus-far collected. Given the environment was just a "previous generation" variation of the BridgeData hardware configuration, I had hoped that the RT-X model would be able to better generalize.

Note that the software captures sufficient data and images necessary to the dataset if it had performed better. 

## REFERENCES:

[1] https://www.researchgate.net/publication/335140409_REPLAB_A_Reproducible_Low-Cost_Arm_Benchmark_for_Robotic_Learning

[2] https://rail-berkeley.github.io/bridgedata/

[3] https://robotics-transformer-x.github.io/

[4] https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb (see image under "First element of the batch returns a robot_net trajectory")

[5] https://docs.google.com/document/d/1si-6cTElTWTgflwcZRPfgHU7-UwfCUkEztkH3ge5CGc/edit

[6] https://github.com/LeninSG21/WidowX

[7] BridgeData V2:
@inproceedings{walke2023bridgedata,
  title={BridgeData V2: A Dataset for Robot Learning at Scale},
  author={Walke, Homer and Black, Kevin and Lee, Abraham and Kim, Moo Jin and Du, Max and Zheng, Chongyi and Zhao, Tony and Hansen-Estruch, Philippe and Vuong, Quan and He, Andre and Myers, Vivek and Fang, Kuan and Finn, Chelsea and Levine, Sergey},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2023}
}

[8] Open X-E:
@misc{open_x_embodiment_rt_x_2023,
title={Open {X-E}mbodiment: Robotic Learning Datasets and {RT-X} Models},
author = {Open X-Embodiment Collaboration and Abhishek Padalkar and Acorn Pooley and Ajinkya Jain and Alex Bewley and Alex Herzog and Alex Irpan and Alexander Khazatsky and Anant Rai and Anikait Singh and Anthony Brohan and Antonin Raffin and Ayzaan Wahid and Ben Burgess-Limerick and Beomjoon Kim and Bernhard Schölkopf and Brian Ichter and Cewu Lu and Charles Xu and Chelsea Finn and Chenfeng Xu and Cheng Chi and Chenguang Huang and Christine Chan and Chuer Pan and Chuyuan Fu and Coline Devin and Danny Driess and Deepak Pathak and Dhruv Shah and Dieter Büchler and Dmitry Kalashnikov and Dorsa Sadigh and Edward Johns and Federico Ceola and Fei Xia and Freek Stulp and Gaoyue Zhou and Gaurav S. Sukhatme and Gautam Salhotra and Ge Yan and Giulio Schiavi and Hao Su and Hao-Shu Fang and Haochen Shi and Heni Ben Amor and Henrik I Christensen and Hiroki Furuta and Homer Walke and Hongjie Fang and Igor Mordatch and Ilija Radosavovic and Isabel Leal and Jacky Liang and Jaehyung Kim and Jan Schneider and Jasmine Hsu and Jeannette Bohg and Jeffrey Bingham and Jiajun Wu and Jialin Wu and Jianlan Luo and Jiayuan Gu and Jie Tan and Jihoon Oh and Jitendra Malik and Jonathan Tompson and Jonathan Yang and Joseph J. Lim and João Silvério and Junhyek Han and Kanishka Rao and Karl Pertsch and Karol Hausman and Keegan Go and Keerthana Gopalakrishnan and Ken Goldberg and Kendra Byrne and Kenneth Oslund and Kento Kawaharazuka and Kevin Zhang and Keyvan Majd and Krishan Rana and Krishnan Srinivasan and Lawrence Yunliang Chen and Lerrel Pinto and Liam Tan and Lionel Ott and Lisa Lee and Masayoshi Tomizuka and Maximilian Du and Michael Ahn and Mingtong Zhang and Mingyu Ding and Mohan Kumar Srirama and Mohit Sharma and Moo Jin Kim and Naoaki Kanazawa and Nicklas Hansen and Nicolas Heess and Nikhil J Joshi and Niko Suenderhauf and Norman Di Palo and Nur Muhammad Mahi Shafiullah and Oier Mees and Oliver Kroemer and Pannag R Sanketi and Paul Wohlhart and Peng Xu and Pierre Sermanet and Priya Sundaresan and Quan Vuong and Rafael Rafailov and Ran Tian and Ria Doshi and Roberto Martín-Martín and Russell Mendonca and Rutav Shah and Ryan Hoque and Ryan Julian and Samuel Bustamante and Sean Kirmani and Sergey Levine and Sherry Moore and Shikhar Bahl and Shivin Dass and Shuran Song and Sichun Xu and Siddhant Haldar and Simeon Adebola and Simon Guist and Soroush Nasiriany and Stefan Schaal and Stefan Welker and Stephen Tian and Sudeep Dasari and Suneel Belkhale and Takayuki Osa and Tatsuya Harada and Tatsuya Matsushima and Ted Xiao and Tianhe Yu and Tianli Ding and Todor Davchev and Tony Z. Zhao and Travis Armstrong and Trevor Darrell and Vidhi Jain and Vincent Vanhoucke and Wei Zhan and Wenxuan Zhou and Wolfram Burgard and Xi Chen and Xiaolong Wang and Xinghao Zhu and Xuanlin Li and Yao Lu and Yevgen Chebotar and Yifan Zhou and Yifeng Zhu and Ying Xu and Yixuan Wang and Yonatan Bisk and Yoonyoung Cho and Youngwoon Lee and Yuchen Cui and Yueh-hua Wu and Yujin Tang and Yuke Zhu and Yunzhu Li and Yusuke Iwasawa and Yutaka Matsuo and Zhuo Xu and Zichen Jeff Cui},
howpublished  = {\url{https://arxiv.org/abs/2310.08864}},
year = {2023},
}

