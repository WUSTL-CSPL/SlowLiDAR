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
    |- testing
        |- calib 
        |- image_2 
        |- velodyne 
```

### PIXOR Model

Please follow the guidelines in [PIXOR Implementation](https://github.com/philip-huang/PIXOR) to install all the dependencies


## Run attack
To run our attack:
```
python run_attack.py --attack_type [perturb or add] --point_idx [point cloud index] --iter_num [number of iterations] --attack_lr [attack learning rate]  --save_path [attack results save path]
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