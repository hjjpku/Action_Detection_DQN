# SAP: Self-Adaptive Proposal Model for Temporal Action Detection based on Reinforcement Learning

By Jingjia Huang, Nannan Li, Tao Zhang and Ge Li

## Introduction

Self-Adaptive Proposal (SAP) is a DQN based model for temporal action localization in untrimmed long videos. The temporal action detection process for SAP is naturally one of observation and refinement: observe the current window and refine the span of attended window to cover true action regions. SAP can learn to find actions through continuously adjusting the temporal bounds in a self-adaptive way.

![Image text](Action_Detection_DQN/img/example.png)

This code has been tested on Ubuntu 16.04 with NVIDIA Tesla K80 of 12GB memory.

## License

SAP is released under the MIT License.

## Citing

If you find SAP useful, please consider citing:

If you like this project, give us a :star: in the github banner :wink:.
