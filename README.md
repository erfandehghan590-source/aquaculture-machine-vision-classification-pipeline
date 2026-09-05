# aquaculture-machine-vision-classification-pipeline
## SalmonScan Classification and Grad-CAM Analysis

This project implements and compares multiple deep learning models for fish health image classification using the SalmonScan dataset.  
It includes standard CNN/Transformer-based classifiers and ProtoNet-based models, with Grad-CAM visualization for infected fish predictions.

## Models
The following models are included:
- MobileNetV3-Large
- ResNet50
- ResNet18
- ViT-B16
- Swin-Tiny
- ConvNeXt-Tiny
- ProtoNet-ResNet50
- ProtoNet-ResNet18
- ProtoNet-ConvNeXt-Tiny
- EfficientNetV2-medium
- DenseNet 121
- YOLOv8n classifier

## Features
- Fish image classification
- Training and evaluation pipeline
- Accuracy and loss logging
- Confusion matrix generation
- Classification report
- Grad-CAM heatmap visualization
- Heatmap saving as image and NumPy array
- Multi-dementional model comparison 

##  Supported Models

| Category | Backbones / Models |
| :--- | :--- |
| **Vision Transformers** | ViT-B/16, Swin-Tiny |
| **Modern & Lightweight CNNs** | ConvNeXt-Tiny, EfficientNetV2-Medium, MobileNetV3-Large, DenseNet-121 |
| **Classical Baselines** | ResNet-18, ResNet-50 |
| **Prototypical Networks (ProtoNet)** | ProtoNet-ResNet18, ProtoNet-ResNet50, ProtoNet-ConvNeXt-Tiny |
| **Real-Time Detection Family** | YOLOv8n-cls |



## 📁 Repository Structure
```text
├── LICENSE
├── .gitignore
├── README.md
├── requirements.txt
├── MVconfig.yaml                      # Global pipeline configuration
├── model_comparison_config.yaml       # Multi-model evaluation settings
├── data_split.py                      # Initial train/val/test splitter
├── data_split_revised.py              # Stratified / updated dataset split
├── common_utils.py                    # Training loops, metrics & logging
├── model_comparison_utils.py          # Benchmark comparison tools
├── proto_specified_utils.py           # ProtoNet distance & loss modules
├── compare_models_script.py           # Comparative evaluation runner
├── run_all_models.py                  # Master batch training script
│
├── ViTB16/                            # ViT-B/16 scripts
├── convnexttiny/                      # ConvNeXt-Tiny scripts
├── mobilenetv3large/                  # MobileNetV3-Large scripts
├── densenet121/                       # DenseNet-121 scripts
├── efficientnetv2m/                   # EfficientNetV2-M scripts
├── resnet18/                          # ResNet-18 scripts
├── resnet50/                          # ResNet-50 scripts
├── swintiny/                          # Swin-Tiny scripts
├── YOLOv8/                            # YOLOv8n classification scripts
├── protonet-with-resnet18-backbone/   # ProtoNet + ResNet-18
├── protonet-with-resnet50-backbone/   # ProtoNet + ResNet-50
├── protonet-with-convnexttiny-backbone/ # ProtoNet + ConvNeXt-Tiny
└── dataset/                           # Dataset directory (Ignored in Git)
    ├── SalmonScan/                    # Raw images
    └── SalmonScan_Split/              # Processed train/val/test splits```
    