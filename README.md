# Psoriasis Detection using CNN with ResNet50

## Project Overview
This project utilizes a Convolutional Neural Network (CNN) with the ResNet50 architecture to detect and classify various skin diseases, including Psoriasis, from dermatological images. The model is trained on a dataset sourced from Kaggle, which includes images of different skin conditions.

## Installation

### Prerequisites
- Python 3.x
- Jupyter Notebook or Google Colab
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/keshavsingh2004/Psoriasis-detection-using-CNN.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Psoriasis-detection-using-CNN
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Open the `Resnet50_skin.ipynb` notebook in Jupyter or Google Colab and run the cells sequentially to train the model and make predictions.

## Data
The dataset can be downloaded from Kaggle. It is automatically downloaded and prepared when you run the notebook.

## Model
The model uses the ResNet50 architecture, pre-trained on ImageNet, and fine-tuned for the task of skin disease classification.

## Results
The model's performance can be evaluated using the metrics printed at the end of the notebook, including accuracy, precision, recall, and F1-score.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
