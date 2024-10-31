# Song Genre Classifier

This repository contains my solution to a DataCamp project on song classification. The project involves building a classifier to predict song genres using machine learning based on various audio features.

## Project Overview

In this project, machine learning techniques are applied to audio data to create a genre classification model. The goal is to predict the genre of a song given its characteristics like tempo, rhythm, and energy.

## Dataset

The dataset includes various audio features that are commonly used in genre classification:
- **Acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
- **Danceability**: How suitable a track is for dancing based on tempo, rhythm, etc.
- **Energy**: Perceptual measure of intensity and activity.
- **Tempo**: Speed of the track.

The dataset is provided by DataCamp as part of their instructional resources.

## Methods

### 1. Data Preprocessing
- **Normalization**: Audio features are scaled to ensure uniform contribution to the model.
- **Encoding**: Categorical data, like genre labels, are encoded for model training.

### 2. Model Selection
- **Logistic Regression**: Used for its interpretability in the initial approach.
- **Support Vector Machine (SVM)**: Applied for enhanced performance with complex boundaries.
- **Model Evaluation**: Models are evaluated based on accuracy, precision, and recall.

## Code Structure

- `main.py`: Contains the main script for loading data, preprocessing, training, and evaluating models.
- `data_preprocessing.py`: Script for cleaning and preparing data.
- `models.py`: Defines and trains the machine learning models.

## Results and Analysis

The final model demonstrates reliable accuracy in predicting song genres based on audio features. Improvements are explored through feature engineering and model tuning.

## Getting Started

### Prerequisites

- Python 3.8+
- Required Libraries: Install dependencies via `pip install -r requirements.txt`.

### Usage

```bash
# Clone the repository
git clone https://github.com/Aleksandar-Mladenoski/song-genre-classifier.git

# Navigate to the project directory
cd song-genre-classifier

# Run the main script
python main.py
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

Thanks to DataCamp for providing the dataset as part of their learning materials. You can check out their datasets at [DataLab](https://www.datacamp.com/datalab/)
