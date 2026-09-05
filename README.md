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

## Project Structure
```text
├──LICENSE
├──.gitignore
├──README.md
├──requirements.txt
├──MVconfig.yaml
├──model_comparison_config.yaml
├──data_split_revised.py
├──data_split.py
├──common_utils.py
├──compare_models_script.py
├──model_comparison_utils.py
├──proto_specified_utils.py
├──run_all_models.py
├──ViTB16
    ├──base-model-VITB16.py
├──convnexttiny
    ├──base-model-convnexttiny.py
├──mobilenetv3large
    ├──base-model-mobilenetv3large.py
├──protonet-with-convnexttiny-backbone
    ├──base-model-protonet-with-convnexttiny-backbone.py
├──protonet-with-resnet18-backbone
    ├──base-model-protonet-with-resnet18-backbone.py
├──protonet-with-resnet50-backbone
    ├──base-model-protonet-with-resnet50-backbone.py
├──resnet18
    ├──base-model-resnet18.py
├──resnet50
    ├──base-model-resnet50.py
├──swintiny
    ├──base-model-swintiny.py
├──densenet121
    ├──base-model-densenet.py
├──efficientnetv2m
    ├──base-model-efficientnet.py
├──YOLOv8
    ├──base-model-YOLOv8.py
└──dataset
    ├──SalmonScan
    └──SalmonScan_Split
```
