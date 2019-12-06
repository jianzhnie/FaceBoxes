# 快速SSD通用物体检测器

机器学习云平台Training pipeline：基于Pytorch框架的快速Faceboxes人脸人体检测

## 特性

- 支持配置基本参数：learning_rate、batch_size、epoch、weight_decay
- 支持网络结构：Inceptionv4
- 支持anchor的选择：anchor size、anchor aspect ratio
- 支持全精度和半精度模型训练
- 支持配置训练集路径

## 环境依赖

-   python=3
-   pytorch=1.0
-   pillow==6.0.0
-   easydict==1.9
-   Cython==0.29.7

## 开始

### 准备数据和标注文件

按照如下目录结构存放训练集的图片数据和标注文件：

```plain
data/
├── annotations
│   ├── 100.xml
│   ├── 101.xml
│   ├── 102.xml
│   ├── 103.xml
│   └── ...
├── images
    ├── 100.jpg
    ├── 101.jpg
    ├── 102.jpg
    ├── 103.jpg
    └── ...

```

其中标准文件格式如下：

```xml
<annotation>
    <folder>VOC2007</folder>
    <filename>100.jpg</filename>
    <source>
        <database>My Database</database>
    </source>
    <size>
        <width>1920</width>
        <height>1080</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>body</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>292</xmin>
            <ymin>518</ymin>
            <xmax>600</xmax>
            <ymax>812</ymax>
        </bndbox>
    </object>
    <object>
        <name>face</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>1161</xmin>
            <ymin>459</ymin>
            <xmax>1215</xmax>
            <ymax>488</ymax>
        </bndbox>
    </object>
</annotation>
```

### 训练示例

训练相关代码均在脚本`train.py`中，以一个人脸人体检测目标为例：
- 最终训练的模型存放于用户给定目录下或者默认目录下
- 训练的迭代次数: 130 epoch
- 训练的初始学习率: 1e-4
- 训练的momentum: 0.95
- 训练的batch size: 16
- 检测的类别数量: 2
- 类别与数字索引的对应关系在xml文件中
- 训练的基础网络: Inceptionv4
- 训练的anchor大小范围是32-128
- 训练的anchor比例是`1:1`
- 训练时支持数据增强操作
- 训练时支持全精度和半精度选择


数据增强和相关参数可在` config.py`中进行配置，如：

```bash
cfg = {
    'name': 'FaceBoxes',
    #'min_dim': 1024,
    #'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'apply_distort': True,
    'apply_expand': True,
    'max_expand_ratio': 1.5,
    'data_anchor_sampling_prob': 0.5,
    'resize_width': 1024,
    'resize_heigth': 1024,
    'min_face_size': 9
}

```

然后调用训练脚本进行训练

```bash
python train.py\
        --batch_size 16 
        --gpunum 4 \ 
        --lr 1e-4 \
        --max_epoch 130 \
        --save_folder student_detector_model \
        --half False
```

#### 帮助文档

训练脚本

```plain
train.py:
  --training_dataset TRAINING_DATASET
                        Training dataset directory
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training
  --num_workers NUM_WORKERS
                        Number of workers used in dataloading
  --gpunum GPUNUM       gpu number
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum MOMENTUM   momentum
  --resume_net RESUME_NET
                        resume net for retraining
  --resume_epoch RESUME_EPOCH
                        resume iter for retraining
  -max MAX_EPOCH, --max_epoch MAX_EPOCH
                        max epoch for retraining
  --weight_decay WEIGHT_DECAY
                        Weight decay for SGD
  --negposratio NEGPOSRATIO
                        negposratio is setted for sample
  --overlap OVERLAP     overlap threshold for match, and negative smaple
  --gamma GAMMA         Gamma update for SGD
  --save_folder SAVE_FOLDER
                        Location to save checkpoint models
  --half HALF           decide to train with half or fp32
  --warmup_epoch WARMUP_EPOCH
                        using warmup to adjust learning rate before the seted
                        epoch

```

### 测试示例

测试相关的代码均在脚本`test.py`中，其中
- 提供两种测试方式：一种是给定测试图像路径，一种是给定测试图像路径列表文件。
- 提供四种文件保存格式：FDDB、WIDER、JSON、VOC。

以人脸人体检测目标为例：

```bash
python test.py\
        --model weights/weights_fbFinal_FaceBoxes.pth\
        --save_folder dmai\
        --outformat WIDER\
        --test_list data/train.txt\
        --half TRUE 
```

#### 帮助文档

```plain
usage: test.py [-h] [-m MODEL] [--save_folder SAVE_FOLDER] [--cpu]
               [--outformat {WIDER,VOC,FDDB,JSON}]
               [--confidence_threshold CONFIDENCE_THRESHOLD] [--top_k TOP_K]
               [--nms_threshold NMS_THRESHOLD] [--keep_top_k KEEP_TOP_K]
               [--test_list TEST_LIST] [--test_dir TEST_DIR] [--half HALF]
               [--face_thresh FACE_THRESH] [--body_thresh BODY_THRESH]

FaceBoxes

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Trained state_dict file path to open
  --save_folder SAVE_FOLDER
                        Dir to save results
  --cpu                 Use cpu inference
  --outformat {WIDER,VOC,FDDB,JSON}
                        outformat
  --confidence_threshold CONFIDENCE_THRESHOLD
                        confidence_threshold
  --top_k TOP_K         top_k
  --nms_threshold NMS_THRESHOLD
                        nms_threshold
  --keep_top_k KEEP_TOP_K
                        keep_top_k
  --test_list TEST_LIST
                        images path list
  --test_dir TEST_DIR   images path dir
  --half HALF           test on half mode or fp32
  --face_thresh FACE_THRESH
                        face_threshold
  --body_thresh BODY_THRESH
                        body_threshold

```

### 测试结果评估

测试相关的代码均在脚本`getEval.py`中，其中
- 结果评估只提供一种数据格式WIDER，xml格式文件可以通过`tools/xml2wider.py`脚本进行转换。
- 结果评估提供PR曲线和目标检测结果误检和漏检结果的可视化
以人脸人体检测目标为例：

```bash
python getEval.py\
        --test_result result.txt\
        --groundtruth groundtruth.txt\
        --save_folder forstudent
```

#### 帮助文档

```plain
usage: getEval.py [-h] [--save_folder SAVE_FOLDER]
                  [--confidence_threshold CONFIDENCE_THRESHOLD]
                  [--iou_threshold IOU_THRESHOLD] [--face_thresh FACE_THRESH]
                  [--body_thresh BODY_THRESH] [--test_result TEST_RESULT]
                  [--groundtruth GROUNDTRUTH]

FaceBoxes

optional arguments:
  -h, --help            show this help message and exit
  --save_folder SAVE_FOLDER
                        Dir to save results
  --confidence_threshold CONFIDENCE_THRESHOLD
                        confidence_threshold
  --iou_threshold IOU_THRESHOLD
                        nms_threshold
  --face_thresh FACE_THRESH
                        face_threshold
  --body_thresh BODY_THRESH
                        body_threshold
  --test_result TEST_RESULT
                        detected results in WIDER format
  --groundtruth GROUNDTRUTH
                        ground truth file in WIDER format

```
