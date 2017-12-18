# two-stream-action-recognition-ucf101
We use a spatial and motion stream cnn with ResNet101 for modeling video information in UCF101 dataset.
## Reference Paper
*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)
*  [[2] Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2)
* [[3] TS-LSTM and Temporal-Inception: Exploiting Spatiotemporal Dynamics for Activity Recognition](https://arxiv.org/abs/1703.10667)

## 1. Data Preprocessing
  ### 1.1 Spatial input data -> rgb frames
  * We extract RGB frames from each video in UCF101 dataset with sampling rate: 10 and save as .jpg image in disk which cost about 5.9G.
  ### 1.2 Motion input data -> stacked optical flow images
  In motion stream, we use two methods to get desire optical flow data. 
  1. Download the preprocessed tvl1 optical flow dataset directly from https://github.com/feichtenhofer/twostreamfusion. 
  2. Using [flownet2.0 method](https://github.com/lmb-freiburg/flownet2-docker) to generate 2-channel optical flow image and save its x, y channel as .jpg image in disk respectively which cost about 56G. 

## 2. Model
  ### 2.1 Spatial cnn
  * As mention before, we use ResNet101 first pre-trained with ImageNet then fine-tuning on our UCF101 spatial rgb image dataset. 
  ### 2.2 Motion cnn
  * Input data of motion cnn is a stack of optical flow images which contained 10 x-channel and 10 y-channel images, So it's input shape is (20, 224, 224) which can be considered as a 20-channel image. 
  * In order to utilize ImageNet pre-trained weight on our model, we have to modify the weights of the first convolution layer pre-trained  with ImageNet from (64, 3, 7, 7) to (64, 20, 7, 7). 
  * In [2] Wang provide a method called **Cross modality pre-training** to do such weights shape transform. He first average the weight value across the RGB channels and replicate this average by the channel number of motion stream input( which is 20 is this case)
  
## 3. Training stategies
  ###  3.1 Spatial cnn
  * Here we utilize the techniques in Temporal Segment Network. for every videos in a mini-batch, we randomly select 3 frames from each video. Then a consensus among the frames will be derived as the video-level prediction for calculating loss.
  ### 3.2 Motion cnn
  * In every mini-batch, we randomly select 64 (batch size) videos from 9537 training videos and futher randomly select 1 stacked optical flow in each video. 
  ### 3.3 Data augmentation
  * Both stream apply the same data augmentation technique such as random cropping.
## 4. Testing method
  * For every 3783 testing videos, we uniformly sample 19 frames in each video and the video level prediction is the voting result of all 19 frame level predictions.
  * The reason we choose the number 19 is that the minimun number of video frames in UCF101 is 28 and we have to make sure there are sufficient frames for testing in 10 stack motion stream.
## 5. Performace
   
 network      | top1  |
--------------|:-----:|
Spatial cnn   | 82.1% | 
Motion cnn    | 80.3% | 
Average fusion| 88.3% |      
   
## 6. Pre-trained Model

* [Spatial resent101](https://drive.google.com/drive/folders/1gVB5StqgoDJ3IxHUn7zoTzTNxzz3du3d?usp=sharing)
* [Motion resent101](https://drive.google.com/drive/folders/1z3fYUOJx_l3BW-NSb7ti0DsyGLFk6Z7J?usp=sharing)

## 7. Testing on Your Device
  ### Motion stream
 Please modify the [funcition of load_ucf_image](https://github.com/jeffreyhuang1/two-stream-action-recognition/blob/master/dataloader/spatial_dataloader.py#L21) to fit the UCF101 dataset on your device
 ### Motion stream
 Please modify this [funcition of stack_opf](https://github.com/jeffreyhuang1/two-stream-action-recognition/blob/master/dataloader/motion_dataloader.py#L32) to fit the UCF101 dataset on your device
 

