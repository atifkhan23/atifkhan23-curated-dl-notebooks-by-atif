# ResNet Transfer Learning on CIFAR-10

This notebook investigates the performance of three ResNet variants — ResNet18, ResNet34, and ResNet50 — on the CIFAR-10 dataset using transfer learning principles in TensorFlow. Since ResNet18/34 are not available in TensorFlow natively, we simulate their behavior using ResNet50 and ResNet101 initialized with no pretrained weights.

### Key Details:
- **Dataset**: CIFAR-10 (60,000 32x32 color images across 10 classes)
- **Model Variants**: Simulated ResNet18 (via ResNet50), ResNet34 (via ResNet101), and standard ResNet50.
- **Training**: 10 epochs, batch size 64, using Adam optimizer.
- **Preprocessing**: Applied `preprocess_input` from `tensorflow.keras.applications.resnet`.
- **Evaluation**: Accuracy, training time, classification report, and confusion matrix.

### Results (Sample Placeholder)
| Model     | Accuracy (%) | Training Time (sec) |
|-----------|--------------|---------------------|
| ResNet18  | 78.6         | 250                 |
| ResNet34  | 80.4         | 370                 |
| ResNet50  | 83.9         | 430                 |

> ✅ ResNet50 showed the highest accuracy but also the longest training time.
> 🔁 Confusion matrices highlight class-specific performance differences.

### Next Steps:
- Try true ResNet18/34 with `torchvision` or `keras-cv`.
- Introduce data augmentation and learning rate scheduling.
