# SAP: Self-Adaptive Proposal Model for Temporal Action Detection based on Reinforcement Learning

By Jingjia Huang, Nannan Li, Tao Zhang and Ge Li

## Introduction

Self-Adaptive Proposal (SAP) is a DQN based model for temporal action localization in untrimmed long videos. The temporal action detection process for SAP is naturally one of observation and refinement: observe the current window and refine the span of attended window to cover true action regions. SAP can learn to find actions through continuously adjusting the temporal bounds in a self-adaptive way.
Experiment results on THUMOS’14 validate the effectiveness of SAP, which can achieve competitive performance with current action detection algorithms via much fewer proposals.

![Image text](https://github.com/hjjpku/Action_Detection_DQN/blob/master/img/framework.png)


![Image text](https://github.com/hjjpku/Action_Detection_DQN/blob/master/img/action.png)


![Image text](https://github.com/hjjpku/Action_Detection_DQN/blob/master/img/example.png)

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


