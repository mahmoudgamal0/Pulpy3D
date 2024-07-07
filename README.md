# Pulpy3D

This code is inspired from the ToothFairy Challenge (MICCAI 2023) and the source code provided in AImageLab [alveolar_canal](https://github.com/AImageLab-zip/alveolar_canal)

The seeding model in Pulpy3D-Seed can be accessed using this [link](https://github.com/mahmoudgamal0/Pulpy3D-Seed)

# Pulp and IAN Segmentation Framework
Pulp and IAN Segmentation Training Framework for 3D CBCT Scans. The framework offers the segmentation tasks through various set of models

* Separate Network for Segmenting either the Pulp or IAN
* Single Semantic Network for Segmenting both the Pulp and IAN
* Multi-Headed Network with a Common Backbone for Segmenting both the Pulp and IAN

## Hardware Setup
| Component | Specification             |
| --------- | ------------------------- |
| CPU       | Intel i7 13700K - 24 core |
| GPU       | Nvidia 4090-RTX 24GB      |
| RAM       | 32 GB DDR4                |
| SSD       | 1TB M2                    |


## Installation
```
# Create python virtual environment
virtualenv venv -p $(which python3)

# Activate virtual environment
source venv/local/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Training
```
python main.py --config configs/segmentation.yml
```

## Dataset
You can access the Pulpy3D dataset using this [link](https://drive.google.com/drive/folders/1M5iU1urLOp1rSxKOm7WCzodAKcZrqT5O?usp=sharing)

The dataset is organized in the following manner
```md
.
├── datasets/
│   └── Pulpy3D/
│       ├── P1/
│       │   ├── data.nii.gz                # CBCT scan 
│       │   ├── gt_pulp.nii.gz             # Semantic Segmentation of Pulp (lower/upper)
│       │   ├── gt_pulp_mandible.nii.gz    # Semantic Segmentation of Pulp (lower only)
│       │   ├── gt_instance.nii.gz         # Instance Segmentation of Pulp
│       │   ├── gt_ian.nii.gz              # IAN Segmentation
│       │   └── gt_instance_ian.nii.gz     # Pulp and IAN Instance Segmentation
│       ├── P2/
│       │   └── ...
│       ├── ...
│       └── splits.json                    # Split file of train, test and val
└── dataset.json                           # Meta data of class labels and mapping
```