# TM2B: Transformer-based Motion-to-Box Network for 3D Single Object Tracking on Point Clouds
Official Implementation for the Paper "TM2B: Transformer-based Motion-to-Box Network for 3D Single Object Tracking on Point Clouds"
## Abstract
<p align="justify">3D single object tracking plays a crucial role in numerous applications such as autonomous driving. Recent trackers based on motion-centric paradigm perform well as they exploit motion cues to infer target relative motion across successive frames, which effectively overcome significant appearance variations of targets and distractors caused by occlusion. However, such a motion-centric paradigm tends to require multi-stage motion-to-box to refine the motion cues, which suffers from tedious hyper-parameter tuning and elaborate subtask designs. In this paper, we propose a novel transformer-based motion-to-box network (TM2B), which employs a learnable relation modeling transformer (LRMT) to generate accurate motion cues without multi-stage refinements. Our proposed LRMT contains two novel attention mechanisms: hierarchical interactive attention and learnable query attention. The former attention builds a learnable number-fixed sampling sets for each query on multi-scale feature maps, enabling each query to adaptively select prominent sampling elements, thus effectively encoding multi- scale features in a lightweight manner, while the latter calculates the weighted sum of the encoded features with learnable global query, enabling to extract valuable motion cues from all available features, thereby achieving accurate object tracking. Extensive experiments demonstrate that TM2B achieves state-of-the-art performance on KITTI, NuScenes and Waymo Open Dataset, while obtaining a significant improvement in inference speed over previous leading methods, achieving 56.8 FPS on a single NVIDIA 1080Ti GPU. 

## Setup

Here, we list the most important part of our dependencies

|Dependency|Version|
|---|---|
|python|3.9.0|
|pytorch|2.0.1|
|mmegine|0.7.4|
|mmcv|2.0.0|
|mmdet|3.0.0|
|mmdet3d|1.1.0|
|spconv|2.3.6|


## Dataset Preparation

### KITTI

+ Download the data for [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
+ Unzip the downloaded files.
+ Put the unzipped files under the same folder as following.
  ```
  [Parent Folder]
  --> [calib]
      --> {0000-0020}.txt
  --> [label_02]
      --> {0000-0020}.txt
  --> [velodyne]
      --> [0000-0020] folders with velodynes .bin files
  ```

### NuScenes

+ Download the dataset from the [download page](https://www.nuscenes.org/download)
+ Extract the downloaded files and make sure you have the following structure:
  ```
  [Parent Folder]
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	        -	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
  ```
>Note: We use the **train_track** split to train our model and test it with the **val** split. Both splits are officially provided by NuScenes. During testing, we ignore the sequences where there is no point in the first given bbox.


### Waymo Open Dataset

+ We follow the benchmark created by [LiDAR-SOT](https://github.com/TuSimple/LiDAR_SOT) based on the waymo open dataset. You can download and process the waymo dataset as guided by [LiDAR_SOT](https://github.com/TuSimple/LiDAR_SOT), and use our code to test model performance on this benchmark.
+ The following processing results are necessary
   ```
    [waymo_sot]
        [benchmark]
            [validation]
                [vehicle]
                    bench_list.json
                    easy.json
                    medium.json
                    hard.json
                [pedestrian]
                    bench_list.json
                    easy.json
                    medium.json
                    hard.json
        [pc]
            [raw_pc]
                Here are some segment.npz files containing raw point cloud data
        [gt_info]
            Here are some segment.npz files containing tracklet and bbox data 
    ```

## Training

+ To train a model, you must specify the `.py` config file. The `.py` file contains all the configurations of the dataset and the model. We provide `.py` config files under the [configs](./configs) directory. 

>Note: Before running the code, you will need to edit the `.py` config file by setting the `path` argument as the correct root of the dataset.

    ```
    # single-gpu training
    python train.py --config $config_file
    # multi-gpu training
    # you will need to edit the `train.py` file by setting the `config` argument
    ./dist_train.sh
    ```

## Testing

+ To test a trained model, specify the checkpoint and the `.py` config file. The `.py` file contains all the configurations of the dataset and the model. We provide `.py` config files under the [configs](./configs) directory.

>Note: Before running the code, you will need to edit the `.py` config file by setting the `path` argument as the correct root of the dataset.

    ```
    # single-gpu testing
    python test.py --config $config_file --load_from $checkpoint_file
    # multi-gpu testing
    # you will need to edit the `test.py` file by setting the `config` and 'load_from' argument
    ./dist_test.sh
    ```
## Acknowledgement
This repo is built upon [Open3DSOT](https://github.com/Ghostish/Open3DSOT) and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). We acknowledge these excellent implementations.
