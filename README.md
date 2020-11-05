
![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg)

![The model architecture](https://img-blog.csdnimg.cn/20201105195218458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01pc3NTaGlob25n,size_16,color_FFFFFF,t_70#pic_center)
<center><b>The Model Struture</b></center><br /><center>Source: https://arxiv.org/</center>
<br /><br />


# Installation
The implementation is based on Python 3.6.2 and Pytorch 1.0.0. You can use the following command to get the environment ready.

```bash
cd path_to_code
pip install -r requirements
```

# Data Preparation
1. Normalization:  This step does 
(a) for each patient, puts multiple modalities(.nii.gz files) and the segmentation label into a into a .npy file. Note that one npy file is roughly 200MB. Please make sure your disk has enough space. 
(b) normalize the MRI images to have mean 0 and std 1

```bash
cd path_to_code
python normalization.py -y 202002
```
Use `-y 2018` or `-y 2020` to specify which dataset(Brats2018 / 2020 training) to be normalized. 202001/202002 indicates Brats2020 Validation/Testing dataset. 

2. Train Test Partition:
Use the script for split. the partition result is store in train_list.txt and valid_list.txt.
```bash
python train_test_split.py
```


# Usage

The Implementations of two stages are separated. 

### Training
- For training the first-stage model, use:
```bash
cd Stage1_VAE
python main.py -e num_epoch -l 128 -g num_gpus -f folder_for_models_saving
``` 

- for training the second-stage model, use:
```bash
cd Stage2_AttVAE
python main_multi.py -e num_epoch -l 128 -g num_gpus -f folder_for_models_saving
``` 
where the `-l` controls for the patch size, which is set to be 128 during our training process. Please change other arguments according to your preference.

### Testing
- For testing the first-stage model, use:
```bash
cd Stage1_VAE
python predict_tta.py -g num_gpus -s folder_to_pth -f first_stage_model_weights.pth
```
- for testing the second-stage model, use the following command:
```bash
cd Stage2_AttVAE
python predict_tta.py -g num_gpus -m folder_for_first-stage_output -f second_stage_model_weights.pth -s folder_to_pth
```
where `-m` sepecifies the segmentation result of which first-stage model will be used to make prediction. 

### Ensemble
You can use our script for model ensemble. 

### Details
Note that the shape of input MRI images need to have 5 dimensions including batch size, with <b>channels-first</b> format. i.e., the shape should look like (bs, c, H, W, D), where:
- `c`, the number of channels are divisible by 4.
- `H`, `W`, `D`, are _all_ divisible by 16.

Please find more details in our paper and code.