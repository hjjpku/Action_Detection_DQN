# To be Updated for AAAI'18

# SAP: Self-Adaptive Proposal Model for Temporal Action Detection based on Reinforcement Learning

By Jingjia Huang, Nannan Li, Tao Zhang and Ge Li

## Introduction

Self-Adaptive Proposal (SAP) is a DQN based model for temporal action localization in untrimmed long videos. The temporal action detection process for SAP is naturally one of observation and refinement: observe the current window and refine the span of attended window to cover true action regions. SAP can learn to find actions through continuously adjusting the temporal bounds in a self-adaptive way.
Experiment results on THUMOS’14 validate the effectiveness of SAP, which can achieve competitive performance with current action detection algorithms via much fewer proposals.

![Image text](https://github.com/hjjpku/Action_Detection_DQN/blob/master/img/framework.png)
fig.1 SAP architecture

![Image text](https://github.com/hjjpku/Action_Detection_DQN/blob/master/img/action.png)
fig.2 Illustration of DQN actions.Each yellow window with dashed lines represents the next window after taking the corresponding action.

![Image text](https://github.com/hjjpku/Action_Detection_DQN/blob/master/img/example.png)
fig.3 Example of how SAP works

This code has been tested on Ubuntu 16.04 with NVIDIA Tesla K80. The CUDA version is 8.0.61

## License

SAP is released under the MIT License.

## Citing

If you find SAP useful, please consider citing:
```
@article{Huang2017A,
  title={A Self-Adaptive Proposal Model for Temporal Action Detection based on Reinforcement Learning},
  author={Huang, Jingjia and Li, Nannan and Zhang, Tao and Li, Ge},
  year={2017},
}
```

If you like this project, give us a :star: in the github banner :wink:.


## Installation:
0. Ensure that you have [gcc](https://gcc.gnu.org/), [torch7](https://github.com/torch/torch7), [CUDA and CUDNN](https://developer.nvidia.com/cuda-downloads).
1. Clone our repo, `git https://github.com/hjjpku/Action_Detection_DQN`
2. Download the pre-trained c3d v1.0 torch model from
    - https://github.com/wandering007/c3d_torch
    - Move the c3d model to our project folder and named it c3d.t7
2. Download pre-trained fc models to our project folder from [BaiduYun]()

## What can you find?
- [Pre-trained DQN models][BaiduYun](). We provide two version of DQN. One for the model w/ temporal pooling and another for the model w/o temporal pooling.
- [Pre-computed action proposals](). Take a look at our results if you are interested in comparisons or building cool algorithms on top of our outputs.

## What can you do with our code?

### Train your own model

- If you want to train your own models on [Thumos'14](http://crcv.ucf.edu/THUMOS14/)

  0. Preproceesing for Thumos dataset
    Down-sample the videos for computing efficiency and save the videos as images. Considering the length of the groundtruth for different action categories varies a lot and the C3D needs an input longer than 16 frames, we down-sample the video of different categories to {1/3;1/3;1;1/2;1;1/3;1/3;1/3;1/3;1/2;1;1/2;1;1/3;1/3;1/3;1/3;1/3;1/3;1/3;1/2} of the original frame rate, respectively.
 
  1. Construct the dataset directory structure as follow:
   ```
    .
    ├── dataset          # put your datasets on the project folder and named it with dataset name
    │   ├── class        # action class name (CleanAndJerk, BaseballPitch, HighJump...)
    │        ├──    
    │        │    ├── clip_idx             # index of videos
    │        │    |   ├── frame_idx.jpg       # video images(from 1 to total sampled image number)
    │        │    |   ...
    │        │    |   ├── frame_idx.jpg
    │        │    |   ├── FrameNum.txt        # total sampled image number
    │        │    |   ├── gt.txt              # groundtruth interval
    │        │    |   ├── NumOfGt.txt         # number of groundtruth in the clip
    │        │    |   └── 
    │        │    ...
    │        │    ├── clip_idx 
    │        │    ├── ClipNum.txt          # number of clips in this category
    │        │    └── 
    │        └── 
    │   ├── class
    |   ...
    |   ├── class
    |   ├── trainlist.t7
    |   ├── validatelist.t7
    └──
    ```
 
  2. Perepare the metadata and annotations for training and testing
    * trainlist.t7 and Thumos_validatelis_new.t7 are the data which indicates the index of videos used for training or validation. It should be a table as follow:
    ```
    {
      1 --class    
      {
         1 --clip_idx
         ...
      }
      ...
      21 --class
      {
        1 --clip_idx 
        ...
      }
    }
    ```
    
    * ClipNum.txt : a single number indicates the number of clips in this category
    
    * FrameNum.txt : a single number indicates the total sampled image number from the clip
    
    * NumOfGt.txt : a single number indicates the number of groundtruth segments in the clip
    
    * gt.txt records the index of first frame and last frame of the groundtruth.It should be arranged as follow:
    ```
    <begin_1> <end_1>
    ...
    <begin_n> <end_n>
    ```
  :wink: If you just want to try out our work and don't want to be bothered by this tedious work, you can download the dataset we have already processed [here]() . 
  
    3. Run the code. For example:
    `th ./Hjj_Training_SAP.lua -data_path Thumos -batch_size 200 -replay_buffer 2000 -lr 1e-3 -class 1` to train the SAP model for first action category in dadaset Thumos.
    If you want to try more simple versions we used for ablation study(SAP w/o temporal pooling; SAP w/o temparal pooling w/o regression), run ./Hjj_Training_DQNRGN.lua or ./Hjj_Trianing_Dqn.lua instead.
    For more details about the script arguments, please consult [Hjj_Read_Input_Cmd.lua](https://github.com/hjjpku/Action_Detection_DQN/blob/master/Hjj_Read_Input_Cmd.lua)
    
    
    

