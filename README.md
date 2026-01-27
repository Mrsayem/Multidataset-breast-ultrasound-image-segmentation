# Multidataset-breast-ultrasound-image-segmentation# Hybrid ResNet-50 for Breast Ultrasound Segmentation

## Overview

This project presents a deep learning approach to automatically segment breast tumors in ultrasound images, distinguishing between benign and malignant lesions. The work combines semantic segmentation with deep supervision to achieve accurate, clinically relevant results that could assist radiologists in breast cancer diagnosis.

### Why This Matters

Breast cancer is one of the leading causes of cancer-related deaths worldwide, and early detection significantly improves patient outcomes. Ultrasound imaging is a critical tool in breast cancer screening, but interpreting these images requires expert knowledge and can be time-consuming. This project aims to develop an automated system that can:

- Accurately identify and segment tumor regions in breast ultrasound images
- Differentiate between benign and malignant tumors
- Provide consistent, reproducible results to support clinical decision-making
- Potentially reduce the workload on radiologists and improve diagnostic accuracy

## Project Highlights

### Key Features

- **Hybrid Architecture**: Built on ResNet-50 with deep supervision at multiple scales
- **Multi-class Segmentation**: Distinguishes between background, benign, and malignant tissue
- **Cross-dataset Validation**: Tested on three different datasets (BUS-UCLM, BUSI, UDIAT) to ensure generalizability
- **Clinical Relevance**: Uses color-coded masks (red for malignant, green for benign) that align with medical conventions
- **Reproducible Results**: Fixed random seeds and deterministic operations for consistent outcomes

### Technical Approach

The model architecture is based on a modified ResNet-50 encoder with a custom decoder that incorporates deep supervision. This means the network learns not just from the final output, but also from intermediate feature maps at multiple resolutions. This approach helps the model capture both fine-grained details and broader context, which is crucial for accurate medical image segmentation.

## Dataset

The primary training dataset is **BUS-UCLM** (Breast Ultrasound - Universidad de Castilla-La Mancha), which contains ultrasound images with pixel-level annotations for benign and malignant tumors. The data includes:

- **38 patients** with breast lesions
- **70% training / 15% validation / 15% test** split
- **Patient-level stratification** to prevent data leakage and ensure fair evaluation
- **Three classes**: Background (black), Benign (green), Malignant (red)

### Additional Validation Datasets

To test the model's ability to generalize beyond the training data, I evaluated it on two external datasets:

1. **BUSI** (Breast Ultrasound Images Dataset): A publicly available dataset with similar breast ultrasound images
2. **UDIAT (DatasetB2)**: Another independent dataset for cross-validation

This cross-dataset testing is crucial because a model that works well only on its training data may not be clinically useful. The ability to perform well on completely different datasets demonstrates the robustness of the approach.

## Model Architecture

### Base Architecture: ResNet-50

The encoder uses ResNet-50, a well-established convolutional neural network that has proven effective in various computer vision tasks. ResNet-50's skip connections help prevent the vanishing gradient problem and allow the network to learn complex patterns efficiently.

### Deep Supervision

One of the key innovations in this project is the incorporation of deep supervision. Instead of only computing the loss at the final output layer, the model is trained with auxiliary losses at multiple intermediate stages:

- **Multiple decoder blocks** at different resolutions (128×128, 64×64, 32×32)
- **Auxiliary loss computation** at each stage, weighted appropriately
- **Better gradient flow** throughout the network, leading to more stable training

This approach helps the model learn hierarchical features more effectively, as it receives feedback at multiple scales during training.

### Loss Function

The training uses a combined loss function:

- **Cross-Entropy Loss** for pixel-wise classification
- **Dice Loss** to handle class imbalance (medical images often have small tumor regions compared to background)
- **Weighted combination** to balance both objectives

## Implementation Details

### Training Configuration

- **Optimizer**: AdamW with weight decay for better generalization
- **Learning Rate**: Initial rate of 1e-4 with cosine annealing scheduler
- **Batch Size**: 8 images per batch
- **Epochs**: 50 with early stopping based on validation performance
- **Data Augmentation**: Random rotations, flips, brightness/contrast adjustments, and elastic transforms
- **Mixed Precision Training**: Using PyTorch's automatic mixed precision for faster training

### Data Preprocessing

Each ultrasound image undergoes several preprocessing steps:

1. **Normalization**: Using ImageNet statistics for consistency with the pretrained ResNet-50
2. **Resizing**: All images standardized to 256×256 pixels
3. **Data Cleaning**: Handling duplicate files and ensuring image-mask correspondence

### Reproducibility

To ensure reproducibility of results, the code includes:

- Fixed random seeds (seed=42) for all random number generators
- Deterministic CUDA operations
- Environment variable settings for reproducible GPU computations
- Detailed documentation of all hyperparameters

## Evaluation Metrics

The model's performance is assessed using several clinical metrics:

### Segmentation Quality

- **Dice Coefficient**: Measures overlap between predicted and ground truth masks (0 to 1, higher is better)
- **Intersection over Union (IoU)**: Another overlap metric commonly used in segmentation tasks

### Classification Performance

Since the task involves distinguishing between background, benign, and malignant tissue:

- **Sensitivity (Recall)**: Ability to correctly identify positive cases (crucial for cancer detection)
- **Specificity**: Ability to correctly identify negative cases
- **Precision**: Proportion of positive predictions that are correct
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Clinical Relevance

The metrics are computed separately for benign and malignant classes, as these have different clinical implications. High sensitivity for malignant tumors is particularly important, as missing a cancer diagnosis can have serious consequences.

## Results Visualization

The project includes comprehensive visualization tools:

1. **Training Curves**: Loss and Dice score progression over epochs
2. **ROC Curves**: For each class to understand classification performance
3. **Confusion Matrices**: To identify common misclassification patterns
4. **Qualitative Results**: Side-by-side comparison of original images, ground truth masks, and predictions

These visualizations help in understanding not just how well the model performs, but also where it succeeds and where it struggles, providing insights for future improvements.

## Technical Stack

### Core Libraries

- **PyTorch**: Deep learning framework
- **torchvision**: For pretrained ResNet-50 and image transformations
- **NumPy**: Numerical computations
- **scikit-learn**: For stratified splitting and metric computation
- **Pillow (PIL)**: Image loading and processing

### Visualization and Analysis

- **Matplotlib**: For plotting training curves and results
- **seaborn**: Enhanced statistical visualizations

### Development Environment

- **Google Colab**: Training on T4 GPU
- **Python 3.x**: Programming language

## File Structure

```
project/
│
├── HybridResnet_50.ipynb    # Main notebook with complete pipeline
├── README.md                 # This file
│
├── data/                     # Datasets (not included in repo)
│   ├── BUS-UCLM/
│   │   ├── images/
│   │   └── masks/
│   ├── BUSI/
│   └── UDIAT/
│
└── outputs/                  # Generated results
    ├── loss_curves.png
    ├── dice_curve.png
    ├── roc_curves.png
    └── confusion_matrix.png
```

## Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy scikit-learn pillow matplotlib seaborn
```

### Running the Code

1. **Mount Google Drive** (if using Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Set up data paths**: Update the paths to your dataset directories in the notebook

3. **Run the notebook cells sequentially**: The notebook is organized in logical sections:
   - Environment setup and reproducibility
   - Dataset loading and preprocessing
   - Model architecture definition
   - Training loop
   - Evaluation and visualization

### Using the Trained Model

To make predictions on new images:

```python
# Load the model
model = HybridResNet50(num_classes=3, deep_supervision=True)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Preprocess your image
image = preprocess(your_image)  # Apply same preprocessing as training

# Make prediction
with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    prediction = torch.argmax(output[0], dim=1)
```

## Challenges and Solutions

### Class Imbalance

**Challenge**: Medical images often have small tumor regions compared to the background, leading to class imbalance.

**Solution**: Combined loss function using both Cross-Entropy and Dice loss, with the Dice component specifically addressing the imbalance by focusing on overlap rather than pixel-wise accuracy alone.

### Limited Data

**Challenge**: Medical datasets are typically small due to privacy concerns and the cost of expert annotations.

**Solution**: 
- Extensive data augmentation to artificially increase dataset diversity
- Transfer learning from ImageNet-pretrained ResNet-50
- Patient-level splitting to prevent overfitting on individual cases

### Generalization Across Datasets

**Challenge**: Models trained on one dataset often perform poorly on different datasets due to variations in imaging protocols and patient populations.

**Solution**: Cross-dataset validation on BUSI and UDIAT to assess true generalization capability and identify potential domain adaptation needs.

## Future Improvements

Based on the current implementation, here are some potential directions for enhancement:

1. **Ensemble Methods**: Combining predictions from multiple models or architectures could improve robustness

2. **Attention Mechanisms**: Incorporating attention modules could help the model focus on clinically relevant regions

3. **Uncertainty Quantification**: Adding uncertainty estimates to predictions could help clinicians understand model confidence

4. **Multi-task Learning**: Simultaneously predicting tumor characteristics (size, shape features) along with segmentation

5. **Domain Adaptation**: Explicitly addressing domain shift when applying the model to different ultrasound machines or protocols

6. **Clinical Integration**: Developing a user-friendly interface for clinical deployment and gathering feedback from radiologists

## Acknowledgments

This project builds upon extensive research in medical image analysis and deep learning. Special thanks to:

- The creators of the BUS-UCLM, BUSI, and UDIAT datasets for making their data publicly available
- The PyTorch team for their excellent deep learning framework
- The research community for advancing the field of medical image segmentation

## Contact and Collaboration

I'm always interested in discussing this work, receiving feedback, or exploring collaboration opportunities. Whether you're a fellow researcher, a clinician, or someone interested in medical AI, I'd love to hear your thoughts.

## License

This project is developed for academic and research purposes. If you use this code or approach in your research, please cite this work appropriately.

---

## Notes on Reproducibility

The code is designed with reproducibility in mind. All random seeds are fixed, and the training process is deterministic on the same hardware. However, note that:

- Results may vary slightly across different GPU architectures
- The exact performance depends on the specific data splits used
- Cross-dataset evaluation is inherently challenging due to domain shift

If you're trying to reproduce these results, ensure you're using the same preprocessing pipeline and hyperparameters as specified in the notebook.

## Ethical Considerations

This is a research project and the model is **not intended for clinical use** without proper validation, regulatory approval, and clinical oversight. Medical AI systems must undergo rigorous testing and validation before being deployed in real clinical settings. This work represents a step toward automated breast cancer detection, but human expert review remains essential for patient care.

---

*This project represents my exploration into applying deep learning to an important medical challenge. I hope it serves as a useful resource for others interested in medical image analysis and contributes to the ongoing efforts to improve breast cancer detection and diagnosis.*
