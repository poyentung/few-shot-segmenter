## Few-shot-segementer

`few-shot-segmenter` is an open-source Python code for complex microstructure segmentation, mainly built with [**Pytorch**](https://pytorch.org/) and [**Scikit-learn**](https://scikit-learn.org/stable/). It is <br />

<div align="center">
    <img src="assets/benchmark_results.png" /> 
</div><br />


## Installation

1. Create a [**Python>=3.9**](https://www.python.org/) environment with [**Conda**](https://docs.conda.io/en/latest/):
```
conda create -n fewshot python=3.9
conda activate fewshot
```

2. Install **Few-shot-segmenter** with [**pip**](https://pypi.org/project/pip/):
```
python -m pip install 'git+https://github.com/poyentung/few-shot-segmenter.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/poyentung/few-shot-segmenter.git
python -m pip install -e few-shot-segmenter
```

## Getting Started

1. **Prepare data and masks**. `demo_data/` folder structures all the necessary images for training and evaluation.

```
# Training data
datapath/
    └── specimen/            
        ├── phase0/
        │   ├── annotation/        # target masks
        │   └── image/             # input images
        ├── phase1/
        │   ├── annotation/
        │   └── image/
        └── ...
```
```
# Training data
datapath/
    ├── query_set/  
    │       ├── iamge1.tiff
    │       ├── iamge2.tiff
    │       └── ...
    ├── (optional) query_ground_truth/                     
    │       ├── phase0/
    │       │   ├── mask1.tiff       
    │       │   ├── mask2.tiff       
    │       │   └── ... 
    │       ├── phase1/
    │       │   ├── mask1.tiff
    │       │   ├── mask2.tiff
    │       │   └── ...
    │       └── ...
    └── support_set/            
            ├── phase0/
            │   ├── annotation/      # target masks
            │   └── image/           # input images
            ├── phase1/
            │   ├── annotation/
            │   └── image/
            └── ...
```

2. **Setup configuration file**. All the config parameters for training modules are saved in the folder [conf/](few-shot-segmenter/conf/), and overidden by `train.yaml` and `train.yaml`. For example, we can set the data augmentation of the datamodule in [train.yaml](few-shot-segmenter/conf/train.yaml):
```
......

datamodule:
    # Configuration
    datapath: ${original_work_dir}/demo_data/train    # directory of training data
    nshot: 3                                          # number of shot for the episodic learning technique
    nsamples: 500                                     # number of images (256*256) cropped from the large image for training

    # hyperparams
    val_data_ratio: 0.15                              # proportion for validation data
    batch_size: 5                                     # batch size for each mini-batch
    n_cpu: 8                                            

    # Data augmentation
    contrast: [0.5,1.5]                               # varying contrast of the image with the boundary condition
    rotation_degrees: 90.0                            # randomly rotate the image up to 90 degree before cropping
    scale: [0.2,0.3]                                  # randomly rescale the image before cropping
    crop_size: 256                                    # crop size of the image for training
    copy_paste_prob: 0.15                             # probability of copy-paste for the training data

......
```

3. **Training**. 
```
cd few-shot-segmenter
python train.py
```

We can also override some of the parameters directly on the commandline. For example,
```
python train.py model_name=test2 datamodule.nshot=5 datamodule.batch_size=10
```

4. **Evaluation**.
We only segment single phase each time we call the function. Please note that this process is GPU-memory-intensive - please reduce the number of `annotator.batch_size` if the relevant error is present. The specified `phase` in the commandline is the filename in the [data folder](few-shot-segmenter/demo_data/train/carbon-chondrite-3slice). For example, if we want to segment `cauliflower` with the model `test` (specified as model_name in the `yaml` file) and a batch_size of 5, we can run:
```
python segment.py model_name=test phase=cauliflower annotator.batch_size=5
```

We can also segment multiple phases in a run:
```
python segment.py --multirun phase=framboid,plaquette,cauliflower
```