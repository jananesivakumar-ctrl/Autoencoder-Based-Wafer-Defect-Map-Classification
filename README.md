# Wafer Defect Classification Using Deep Learning

Comparing convolutional neural networks (CNNs), autoencoder-based augmentation, and MobileNetV2 transfer learning for semiconductor wafer defect classification.

## Overview

Automated wafer inspection helps identify defects that affect semiconductor manufacturing yield. This project investigates how severe class imbalance affects deep learning models trained on low-resolution wafer maps, and whether data augmentation or transfer learning improves the detection of rare defects.

Developed by **Janane Sivakumar** for **EE 267: Computer Vision with AI Applications** at **San José State University**.

## Dataset

The experiments use **14,366 wafer-map images**, each with a spatial resolution of **26 × 26 pixels**, across eight defect types and one defect-free class.

| Class | Number of images |
| --- | ---: |
| None (defect-free) | 13,489 |
| Loc | 297 |
| Edge-Loc | 296 |
| Center | 90 |
| Random | 74 |
| Scratch | 72 |
| Edge-Ring | 31 |
| Near-full | 16 |
| Donut | 1 |

Approximately **94%** of the images belong to the defect-free class, making overall accuracy alone an insufficient measure of defect detection performance. The experiments use a **70% training, 15% validation, and 15% testing** split.

## Models Compared

| Approach | Method |
| --- | --- |
| Baseline CNN | Compact CNN trained on the original imbalanced dataset. |
| Augmented CNN | CNN incorporating minority-class oversampling, autoencoder reconstructions, class weighting, and a validation subset with capped samples per class. |
| Transfer learning | ImageNet-pretrained MobileNetV2, first trained with a frozen backbone and then partially fine-tuned. |

Performance is evaluated using training and validation curves, confusion matrices, and per-class precision, recall, and F1-scores.

## Key Results

- **Baseline CNN:** Predominantly predicted the defect-free class, achieving high overall accuracy while failing to identify minority defects.
- **Augmented CNN:** Improved recognition of several minority classes, with test-set recall of **80% for Edge-Ring**, **75% for Random**, and **50% for Scratch**. This improvement came with more false positives and lower accuracy on defect-free wafers.
- **MobileNetV2:** Did not improve minority-class detection in these experiments, despite high overall accuracy and additional fine-tuning.

These results highlight the importance of evaluating individual defect classes and addressing data imbalance when developing inspection models.

## How to Run

The project was developed using **Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, and Seaborn** in **Google Colab**, using CPU-based hardware.

1. Download a project notebook (`.ipynb`) from this repository and open it in Google Colab.
2. Obtain the dataset file, `waferImg26x26.pkl`, separately if it is not included in the repository.
3. Upload the dataset to the Colab runtime or place it in your Google Drive.
4. Update `DATA_PKL` in the notebook to match the dataset location. Mount Google Drive if using a Drive path.
5. Run the notebook cells in order to preprocess the data, train the model, and generate evaluation plots.

If the required packages are missing, run the following in a Colab cell:

```python
%pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

## Limitations and Future Work

The small number of minority samples, low image resolution, and limited diversity of autoencoder reconstructions constrain performance. Reported results come from a fixed dataset split; some rare classes have very few test examples, and the Donut class has no validation or test examples in the reported split.

Future work could evaluate repeated data splits, collect more diverse defect examples, use higher-resolution wafer maps, and explore alternative augmentation methods or imbalance-aware loss functions.

## Project Report

The full report provides model architectures, preprocessing details, training curves, confusion matrices, and discussion:

[Read the project report](EE%20267-Mini-Project%20%232%20%281%29.pdf)

This link assumes the PDF retains its original filename and is uploaded alongside this README. Update the link if the report is renamed or moved into a folder.
