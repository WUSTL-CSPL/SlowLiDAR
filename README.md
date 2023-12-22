# SlowLiDAR: Increasing the Latency of LiDAR-Based Detection Using Adversarial Examples

## Install

### KITTI datasets

Please download the following data: [point cloud](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip), [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip), [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip), [labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip), and re-organize the datasets as follows:
```
kitti
    |- training
        |- calib 
        |- image_2 
        |- label_2 
        |- velodyne 
    |- validation
        |- calib 
        |- image_2 
        |- velodyne 
    |- train.txt
    |- val.txt
```
Your can also download our organized KITTI dataset from [here](https://drive.google.com/file/d/1Lrp5ncbqu07ChMrQCvXndSTHmKLDm8A-/view?usp=sharing).

Also, please change the KITTI_PATH in datagen.py to the above kitti folder. 

### Software Specification
Cuda Version: 11.7

Python Version: 3.9.7

Python Environments: please see environment.yml. You can install all of them by the following command:

```
conda env create -f environment.yml
```

### Hardware Specification

Please see hardware_spec.txt.

### PIXOR Model

Please follow the guidelines in [PIXOR Implementation](https://github.com/philip-huang/PIXOR) to install all the dependencies. You can skip this step if you use the above conda command to install the environments.

#### Changes
Compared with original [PIXOR implementations](https://github.com/philip-huang/PIXOR), we have make the following changes:

- We have re-implemented the pre-processing pipelines in Python, transitioning from the original C++ implementation. Detailed descriptions of this process are available in our paper. 

- We have changed the 'nms_top' parameters in this repository, increasing the proposal cap from the previously hard-coded limit of 64 to 5000 to comprehensively assess the effects of our attacks. More discussions on the impact of this parameter are in the ablation study of our paper.

## Run attack
To run perturbation-based attack:
```
python run_attack.py --attack_type perturb --point_idx 10 --iter_num 2000 --attack_lr 0.01  --save_path ./results/
```

To run addition-based attack:
```
python run_attack.py --attack_type add --point_idx 10 --iter_num 2000 --attack_lr 0.01  --save_path ./results/
```
## Evaluation
To evaluate the running time of the original sample and adversarial sample, we prepared one script to run the inference and print out the time. Also, we have prepared two of our pre-trained samples.

To evaluate the original point cloud:
```
python gpu_time_benchmark.py --eval_type ori --point_idx 10
```

To evaluate the perturbation adversarial example: 
```
python gpu_time_benchmark.py --eval_type adv --path ./adv_examples/final_perb.pt
```

To evaluate the addition adversarial example: 
```
python gpu_time_benchmark.py --eval_type adv --path ./adv_examples/final_add.pt
```



## Citation
If you find our work useful, please cite:

```
@InProceedings{Liu_2023_CVPR,
    author    = {Liu, Han and Wu, Yuhao and Yu, Zhiyuan and Vorobeychik, Yevgeniy and Zhang, Ning},
    title     = {SlowLiDAR: Increasing the Latency of LiDAR-Based Detection Using Adversarial Examples},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {5146-5155}
}
```

## Acknowledements

Thanks for the open souce code https://github.com/philip-huang/PIXOR
