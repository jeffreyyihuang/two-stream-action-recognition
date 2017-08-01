# pytorch-two-stream-cnn-ucf101
We use a spatial and motion stream cnn with ResNet101 as baseline for modeling video information in UCF101 dataset.

## 1. Data preprocessing
  ### 1.1 Spatial input data -> rgb frames
  We extract RGB frames from each video in UCF101 dataset with sampling rate: 10 and save as .jpg image in disk which cost about 5.9G.
  ### 1.2 Motion input data -> stacked optical flow images
  In motion stream, we use two methods to get desire optical flow data. First is to download the preprocessed tvl1 optical flow dataset directly from https://github.com/feichtenhofer/twostreamfusion. Second method is use [flownet2.0](https://github.com/lmb-freiburg/flownet2-docker) to generate 2-channel optical flow image and save its x,y channel as .jpg image in disk which cost about 56G 
    
