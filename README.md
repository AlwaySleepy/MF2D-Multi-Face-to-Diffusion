# 本项目是基于Face2Diffusion论文完成的一个复现&改进项目

## 项目概览
```text
.
├── &
├── arcface_torch               # msid training
│   ├── backbones
│   │   ├── msid.py
│   │   └── ...
│   ├── configs
│   │   ├── msid_config.py
│   │   └── ...
│   ├── train_v2.py
│   └── ...
├── checkpoints                 # msid ckpt and fmap ckpt
│   ├── mapping.pt
│   ├── msid.pt
│   └── ...
├── dataori.py                  # data preprocess for MF2D
├── data.py                     # data preprocess for cgdr
├── F2D_inference_mono.py       # F2D infer mono subject
├── F2D_inference_multi.py      # F2D infer multi subject
├── f2d_train.py                # F2D training with cgdr
├── fmap                        # MF2D fmap ckpt
├── MF2D_inference.py           # MF2D infer multi subject
├── MF2D_train.py               # MF2D training
├── model.py                    
├── transform.py
└── ...
```

## 论文复现

* __msid复现:__
    * 环境搭建: 按照[face2diffusion](https://github.com/mapooon/Face2Diffusion)主页的环境进行搭建, 在docker内运行项目.
	* 具体实现: 下载[arcface_torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)内提供的[MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57)数据集MS1M-ArcFace, 
    放到arcface_torch/data路径下, 然后运行
        ```text
        python train_v2.py configs/msid_config
        ```
        即可实现单gpu训练msid encoder.

* __cgdr复现:__
	* 环境搭建: 由于face2diffusion没有给出官方的训练代码, 所以我们主要参考了类似的项目[fastcomposer](https://github.com/mit-han-lab/fastcomposer)自己写了一份训练代码. 环境的搭建参考fastcomposer主页.
	* 具体实现: 
    
        下载ffhq数据集放入data路径下
  
        ```bash 
        cd data
        wget https://huggingface.co/datasets/mit-han-lab/ffhq-fastcomposer/resolve/main/ffhq_fastcomposer.tgz
        tar -xvzf ffhq_fastcomposer.tgz
        ```
        修改data.py文件内collate_fn内的cgdr参数为True, 然后运行
        ```text
        python f2d_train.py
        ```





## 我们的改进: Multi Face to Diffusion
Face2Diffusion本身是一个用比较灵巧的trick来提升个性化人脸生成效果的项目, 但是其局限性较强——只针对单一主体的生成. 我们将其架构简单地改成了双主体的架构, 利用现有的ckpt尝试直接生成多主体图片, 发现其效果非常不理想. 


我们在调查资料的时候发现了另一个与多主体有关的库: fastcomposer, 因此我们觉得可以尝试将二者的思想结合, 将face2diffusion项目的思想推广到多主体生成中, 让face2diffusion能够具备不错多主体生成能力. 


### 改进方法: 
针对多主体数据, 训练相应的Fmap. 使用FastComposer数据集, 将训练图像首先通过物体分割模型, 得到多个主体的图像, 然后使用F2D的MSID对多个图像进行编码, 将图像特征通过Fmap转化成文本空间的向量, 插入文本描述的对应位置中得到增强的文本向量, 然后送入unet进行去噪流程, 冻结MSID与unet参数. 
相关ckpt存储在fmap/路径下.

* __训练方法:__
    * 下载ffhq数据集, 参考cgdr复现
    * 下载[face2diffusion](https://github.com/mapooon/Face2Diffusion)提供的msid以及fmap ckpt, 放入checkpoints路径下
    * 运行以下命令: 
        ```text
        python MF2D_train.py
        ```
* __inference对比__
    * 准备两张图放入data/input/路径下(已提供一些图, 可以自行替换)
    * 如果想运行MF2D, 运行以下命令
        ```text
        python MF2D_inference.py
        ```
    * 如果想运行F2D, 运行以下命令
        ```text
        python F2D_inference_mono.py    # for one subject
        python F2D_inference_multi.py   # for multi subject
        ```
    * 可在上述文件中自行修改prompt, 实现不同输出.

