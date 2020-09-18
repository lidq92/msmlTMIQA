# Quality Assessment for Tone-Mapped HDR Images Using Multi-Scale and Multi-Layer Information
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](License)

## Description
Code for the following paper:

- Qin He, Dingquan, Tingting Jiang, and Ming Jiang. "[Quality Assessment for Tone-Mapped HDR Images Using Multi-Scale and Multi-Layer Information](https://www.researchgate.net/publication/328418186_Quality_Assessment_for_Tone-Mapped_HDR_Images_Using_Multi-Scale_and_Multi-Layer_Information)." ICMEw, 2018.
## Requirement
Framework: [Caffe](https://github.com/BVLC/caffe/) 1.0 (with CUDA 8.0) + [MATLAB](https://www.mathworks.com/products/matlab.html) 2016b Interface

Download the ResNet-50-model.caffemodel from https://github.com/KaimingHe/deep-residual-networks and paste it into the directory "models/" before using the code! 
It's about 100MB which is too large to upload to this repo.
If you have difficulty, you can also download the `ResNet-50-model.caffemodel` in [my sharing on BaiduNetDisk](https://pan.baidu.com/s/1T32sYjrQA04kl1auArirxw) with password `u8sd`.

## Feature Extraction
The features are extracted from the DCNN models pre-trained on the image classification task.

Remember to change the value of "im_dir" and "im_lists" in data info！

Run `ExtractFeatures.m` to get the features. For features of images from the ESPL-LIVE HDR dataset, you can also download from [my sharing on BaiduNetDisk](https://pan.baidu.com/s/1lgGRTNEG_JwL_uHm7mpFzg) with password `3aj0`.

## Quality Prediction by PLSR
All we need to train is a PLSR model, where the training function is plsregress.m in [MATLAB](https://www.mathworks.com/products/matlab.html). 

Run `QualityPrediction.m` to conduct the experiments on ESPL-LIVE HDR.

## Citation

Please cite our paper if it helps your research:

<pre><code>@inproceedings{he2018quality,
  title={Quality Assessment for Tone-Mapped HDR Images Using Multi-Scale and Multi-Layer Information},
  author={He, Qin and Li, Dingquan and Jiang, Tingting and Jiang, Ming},
  booktitle={ICMEw},
  year={2018}
}</code></pre>

## Contact
Dingquan Li, dingquanli@pku.edu.cn.
