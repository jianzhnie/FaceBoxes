# FaceBoxes in PyTorch

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

A [PyTorch](https://pytorch.org/) implementation of [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234). The official code in Caffe can be found [here](https://github.com/sfzhang15/FaceBoxes).

## Performance
| Dataset | Original Caffe | PyTorch Implementation |
|:-|:-:|:-:|
| AFW | 98.98 % | 98.55% |
| PASCAL | 96.77 % | 97.05% |
| FDDB | 95.90 % | 96.00% |


### Contents
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [References](#references)

## Installation
1. Install [PyTorch](https://pytorch.org/) >= v1.0.0 following official instruction.

2. Clone this repository. We will call the cloned directory as `$FaceBoxes_ROOT`.
```Shell
git clone https://github.com/zisianw/FaceBoxes.PyTorch.git
```

3. Compile the nms:
```Shell
./make.sh
```

_Note: Codes are based on Python 3+._

## Training
1. Download [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html) dataset, place the images under this directory:
  ```Shell
  $FaceBoxes_ROOT/data/WIDER_FACE/images
  ```
2. Convert WIDER FACE annotations to VOC format or download [our converted annotations](https://drive.google.com/open?id=1-s4QCu_v76yNwR-yXMfGqMGgHQ30WxV2), place them under this directory:
  ```Shell
  $FaceBoxes_ROOT/data/WIDER_FACE/annotations
  ```
  You can use the convert tools in $FaceBoxes_ROOT/wider-face-pascal-voc-annotations, run:

  ```shell
  python  convert.py
  ```

3. 为了能够检测小目标， 这里去掉了对于 size <16 的 box 过滤，同时增加了Priors 对于 size= 16 的密集采样

- modify layers/function/prior_box.py

  ```python 
  if min_size == 16:
      dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
      dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
      for cy, cx in product(dense_cy, dense_cx):
          anchors += [cx, cy, s_kx, s_ky]
  elif min_size == 32:
      dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
      dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
      for cy, cx in product(dense_cy, dense_cx):
          anchors += [cx, cy, s_kx, s_ky]
  ```
- comment the following lines in  data/data_augment.py

  ```python
  # make sure that the cropped image contains at least one face > 16 pixel at training image scale
  # b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
  # b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
  # mask_b = np.minimum(b_w_t, b_h_t) > 16.0
  # boxes_t = boxes_t[mask_b]
  # labels_t = labels_t[mask_b]
  ```
- add loc and cls predict head in models/faceboxes.py 

  ```python 
  loc_layers += [nn.Conv2d(128, 37 * 4, kernel_size=3, padding=1)]
  conf_layers += [nn.Conv2d(128, 37 * num_classes, kernel_size=3, padding=1)]
  ```

4. Train the model using WIDER FACE:
  ```Shell
  cd $FaceBoxes_ROOT/
  python train.py
  
  python main.py
  ```

If you do not wish to train the model, you can download [our pre-trained model](https://drive.google.com/file/d/1tRVwOlu0QtjvADQ2H7vqrRwsWEmaqioI) and save it in `$FaceBoxes_ROOT/weights`.


## Evaluation
1. Download the images of [AFW](https://drive.google.com/open?id=1Kl2Cjy8IwrkYDwMbe_9DVuAwTHJ8fjev), [PASCAL Face](https://drive.google.com/open?id=1p7dDQgYh2RBPUZSlOQVU4PgaSKlq64ik) and [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to:
```Shell
$FaceBoxes_ROOT/data/AFW/images/
$FaceBoxes_ROOT/data/PASCAL/images/
$FaceBoxes_ROOT/data/FDDB/images/
```

2. Evaluate the trained model using:
```Shell
# dataset choices = ['AFW', 'PASCAL', 'FDDB']
python3 test.py --dataset FDDB
# evaluate using cpu
python3 test.py --cpu
# visualize detection results
python3 test.py -s --vis_thres 0.3
```

3. Download [eval_tool](https://bitbucket.org/marcopede/face-eval) to evaluate the performance.
    
## References
- [Official release (Caffe)](https://github.com/sfzhang15/FaceBoxes)
- A huge thank you to SSD ports in PyTorch that have been helpful:
  * [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [RFBNet](https://github.com/ruinmessi/RFBNet)

  _Note: If you can not download the converted annotations, the provided images and the trained model through the above links, you can download them through [BaiduYun](https://pan.baidu.com/s/1HoW3wbldnbmgW2PS4i4Irw)._
