# Bug Detection in Code Using Machine Learning

An intelligent system that automatically detects potential bugs in code snippets using machine learning and natural language processing techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Version 2: Improved Model](#-version-2-improved-model)
- [Dataset](#dataset)
- [Technical Details](#technical-details)
- [Limitations & Future Work](#limitations--future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a machine learning classifier that analyzes code snippets and predicts whether they contain bugs. The system uses TF-IDF vectorization to extract features from code and trains a Logistic Regression model to distinguish between buggy and correct code.

**Key Capabilities:**
- Automated bug detection in code snippets
- Binary classification (buggy vs. correct)
- Scalable preprocessing pipeline
- Easy-to-use prediction interface

## ✨ Features

- **Automated Data Processing**: Convert raw code files into ML-ready format
- **TF-IDF Vectorization**: Extract meaningful features from code syntax
- **Model Persistence**: Save and load trained models for reuse
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, and recall
- **Simple Prediction API**: Easy integration into existing workflows
- **Version Control**: Multiple model versions with performance tracking

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/KaunAnkit/bug-detection-ml.git
   cd bug-detection-ml
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
joblib>=1.0.0
```

## ⚡ Quick Start

### Using v1 (Baseline Pipeline)

Run the complete pipeline with these commands:

```bash
# 1. Preprocess the raw data
python src/data_preprocessing.py

# 2. Extract features using TF-IDF
python src/feature_extraction.py

# 3. Train the model
python src/model_training.py

# 4. Evaluate performance
python src/model_evaluation.py

# View the results
cat models/evaluation_report.txt
```

### Using v2 (Improved Model)

```bash
# Run the improved training pipeline in Jupyter
jupyter notebook src_v2/model_training_v2.ipynb

# Or view the saved results
cat models/evaluation_report_v2.txt
```

## 📁 Project Structure

```
bug-detection-ml/
├── data/
│   ├── processed/
│   │   └── cleaned_data.csv      # Preprocessed and labeled dataset
│   └── raw/
│       ├── bug/                  # Buggy code examples
│       └── correct/              # Correct code examples
│
├── models/
│   ├── evaluation_report.txt    # v1 model performance metrics
│   ├── evaluation_report_v2.txt # v2 model performance metrics (NEW)
│   ├── trained_model.pkl        # v1 Logistic Regression classifier
│   ├── trained_model_v2.pkl     # v2 Improved classifier (NEW)
│   ├── vectorizer.pkl           # TF-IDF vectorizer
│   ├── X_Tfidf.pkl             # Transformed feature matrix
│   └── y.pkl                   # Target labels
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning and CSV generation
│   ├── feature_extraction.py    # TF-IDF vectorization and feature creation
│   ├── model_training.py        # Logistic Regression model training
│   └── model_evaluation.py      # Performance evaluation and reporting
│
├── src_v2/
│   └── model_training_v2.ipynb  # Improved model training pipeline (NEW)
│
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## 📖 Usage Guide

### Complete Pipeline Execution (v1)

Run each script in sequence:

```bash
# Step 1: Preprocess raw code files into cleaned CSV
python src/data_preprocessing.py

# Step 2: Extract TF-IDF features from cleaned data
python src/feature_extraction.py

# Step 3: Train the Logistic Regression model
python src/model_training.py

# Step 4: Evaluate model and generate report
python src/model_evaluation.py
```

### Working with Your Own Data

1. **Add code samples** to the raw data folders:
   - Place buggy code files in `data/raw/bug/`
   - Place correct code files in `data/raw/correct/`

2. **Run the complete pipeline**:
   ```bash
   python src/data_preprocessing.py
   python src/feature_extraction.py
   python src/model_training.py
   python src/model_evaluation.py
   ```

### Viewing Results

After running the evaluation script, check the performance report:

```bash
# v1 results
cat models/evaluation_report.txt

# v2 results
cat models/evaluation_report_v2.txt
```

The report contains:
- Overall accuracy score
- Confusion matrix
- Detailed classification report (precision, recall, F1-score for each class)

## 📊 Model Performance

### Version 1 (Baseline)

Initial model metrics on the test set:

| Metric | Value |
|--------|-------|
| **Accuracy** | 55.56% |
| **Precision (Buggy Code)** | 0.62 |
| **Recall (Buggy Code)** | 0.83 |
| **F1-Score (Buggy Code)** | 0.71 |

**Confusion Matrix**

```
                Predicted
              Buggy  Correct
Actual Buggy   [5]     [1]
     Correct   [3]     [0]
```

**Note**: The v1 model shows higher recall (83%) than precision (62%), meaning it's better at catching bugs but produces false positives.

## 🚀 Version 2: Enhanced Model with Advanced Features

A second version of the bug detection pipeline was developed and trained in Google Colab, achieving **significant performance improvements** through advanced algorithms and engineered features with a **12.2% accuracy increase**.

### 🎯 Performance Comparison

| Metric | v1 (Baseline) | v2 (Logistic Regression) | v3 (XGBoost + Features) | Best Improvement |
|--------|---------------|--------------------------|-------------------------|------------------|
| **Accuracy** | 55.56% | 67.83% | **TBD** | **+12.27%** |
| **Precision (Buggy)** | 0.62 | 0.91 | **TBD** | **+46.8%** |
| **Recall (Buggy)** | 0.83 | 0.40 | **TBD** | -51.8% |
| **F1-Score (Buggy)** | 0.71 | 0.55 | **TBD** | -22.5% |
| **Precision (Correct)** | - | 0.61 | **TBD** | - |
| **Recall (Correct)** | - | 0.96 | **TBD** | - |
| **F1-Score (Correct)** | - | 0.75 | **TBD** | - |

### 📈 v2 Detailed Metrics

**Overall Accuracy: 67.83%**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0 (Buggy)** | 0.91 | 0.40 | 0.55 | 2,366 |
| **1 (Correct)** | 0.61 | 0.96 | 0.75 | 2,374 |
| **Macro Avg** | 0.76 | 0.68 | 0.65 | 4,740 |
| **Weighted Avg** | 0.76 | 0.68 | 0.65 | 4,740 |

**Confusion Matrix**

```
                Predicted
              Buggy  Correct
Actual Buggy   [937]  [1429]
     Correct   [96]   [2278]
```

### 🔍 Model Interpretation

**Strengths:**
- **High Precision for Buggy Code (0.91)**: When the model predicts code is buggy, it's correct 91% of the time
- **Excellent Recall for Correct Code (0.96)**: Catches 96% of all correct code samples
- **12% Higher Accuracy**: More reliable overall predictions compared to v1
- **Balanced Performance**: Better weighted average metrics (0.76 precision, 0.68 recall)

**Trade-offs:**
- Lower recall for buggy code (0.40) means it misses 60% of actual bugs
- More conservative in flagging bugs, reducing false positives significantly
- Better suited for applications where false positives are costly

**Use Cases:**
- **v1 Model**: Best when you need to catch as many bugs as possible (high recall)
- **v2 Model**: Best when you need confidence in bug predictions (high precision)

### ⚙️ v2 Model Details

**Multiple Model Architectures Tested:**

1. **Logistic Regression (Baseline v2)**
   - Algorithm: Logistic Regression with optimized hyperparameters
   - Accuracy: 67.83%
   
2. **Random Forest Classifier**
   - Algorithm: Ensemble of 100 decision trees
   - Parameters: `n_estimators=100, max_depth=5`
   - Performance: Evaluated for comparison

3. **XGBoost Classifier (Standard)**
   - Algorithm: Gradient Boosting
   - Parameters: `n_estimators=100, max_depth=6, learning_rate=0.1`
   - Performance: Evaluated for comparison

4. **XGBoost with Engineered Features (Advanced)**
   - Algorithm: XGBoost + Custom code features
   - Enhanced Feature Set:
     - **Control Flow Features**: `for` loop count, `if` statement count
     - **Function Metrics**: Return statement count, function definition count
     - **Code Quality**: Comment count, try-except presence
     - **Structural Features**: Average indentation depth
   - TF-IDF Parameters: `ngram_range=(1,3), max_features=100000, min_df=2`
   - Combined Features: TF-IDF vectors + 7 engineered features
   - Performance: Evaluated for comparison

**Vectorization Details:**
- **TF-IDF Vectorizer**: Expanded to capture more patterns
- **n-gram range**: (1,3) - captures single words, pairs, and triplets
- **Max features**: 100,000 (increased from 500)
- **Min document frequency**: 2 (reduces noise)

**Training Configuration:**
- Dataset: 4,740 samples (expanded and balanced)
- Split: 80% train, 20% test
- Random state: 42 (for reproducibility)
- Environment: Google Colab (GPU-accelerated)

### 📁 v2 File Locations

```
src_v2/
└── model_training_v2.ipynb      # Improved training pipeline

models/
├── trained_model_v2.pkl         # Saved v2 model
└── evaluation_report_v2.txt     # v2 performance report
```

### 🧠 Run v2 Locally

To reproduce or fine-tune the improved model:

```bash
# Activate environment and install dependencies
pip install -r requirements.txt

# Open and run the v2 notebook
jupyter notebook src_v2/model_training_v2.ipynb
```

## 💾 Dataset

### Data Source

The training dataset is **AI-generated** due to the scarcity of publicly available, labeled bug datasets. 

- **v1 Dataset**: 9 samples (proof of concept)
- **v2 Dataset**: 4,740 samples (expanded for production readiness)

### Common Bug Types Included

- Off-by-one errors
- Null pointer/None reference errors
- Type mismatches
- Logic errors
- Index out of bounds
- Infinite loops
- Resource leaks

### Data Format

```csv
Code_text,label
"def add(a, b): return a + b",1
"def divide(a, b): return a / b",0
```

**Labels**: 
- `0` = Buggy code
- `1` = Correct code

## 🔬 Technical Details

### Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)**
- Converts code into numerical feature vectors
- Captures keyword importance and code patterns
- Parameters: `max_features=500, ngram_range=(1,2)`

### Model Architecture

**Logistic Regression Classifier**
- Fast training and inference
- Interpretable coefficients
- Good baseline for text classification
- v1 Parameters: `max_iter=1000, random_state=42`
- v2 Parameters: Optimized hyperparameters via grid search

### Training Process

1. **Data Preprocessing** (`data_preprocessing.py`):
   - Reads code files from `data/raw/bug/` and `data/raw/correct/`
   - Creates labeled dataset with columns: Code_text, label
   - Saves to `data/processed/cleaned_data.csv`

2. **Feature Extraction** (`feature_extraction.py`):
   - Loads cleaned data
   - Applies TF-IDF vectorization to code text
   - Saves vectorizer and transformed features as pickle files

3. **Model Training** (`model_training.py` / `model_training_v2.ipynb`):
   - Loads TF-IDF features and labels
   - Splits data (80% train, 20% test, random_state=42)
   - Trains Logistic Regression classifier
   - Saves trained model

4. **Model Evaluation** (`model_evaluation.py` / within v2 notebook):
   - Loads trained model and test data
   - Generates predictions
   - Calculates metrics and saves evaluation report

## ⚠️ Limitations

### Current Limitations

1. **Dataset Origin**: AI-generated dataset may not reflect all real-world bug patterns
2. **Recall-Precision Trade-off**: v2 prioritizes precision over recall for buggy code
3. **Syntax-Only Analysis**: Doesn't understand code semantics or execution flow
4. **Language-Specific**: Currently optimized for Python code only
5. **Model Simplicity**: Logistic Regression may not capture complex bug patterns

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution

- Adding more diverse training data
- Implementing new feature extraction techniques
- Experimenting with different ML models
- Improving documentation
- Writing unit tests
- Creating visualization tools
- Developing ensemble methods combining v1 and v2

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Ankit Jha - [LinkedIn](www.linkedin.com/in/connectwithankitjha) - connectwithankitjha@gmail.com

Project Link: [https://github.com/KaunAnkit/bug-detection-ml](https://github.com/KaunAnkit/bug-detection-ml)

## 🙏 Acknowledgments

- Inspired by research in automated program repair and static analysis
- TF-IDF implementation from scikit-learn
- Code examples generated with assistance from AI tools
- Google Colab for GPU-accelerated training infrastructure

---

**⚠️ Disclaimer**: This is a research and educational project. The model should not be used as the sole method for bug detection in production systems. Always combine automated tools with manual code review and comprehensive testing.

---

**🎯 Quick Stats**
- ⭐ v2 Model Accuracy: **67.83%**
- 📈 Improvement over v1: **+12.27%**
- 🎯 Buggy Code Precision: **91%**
- ✅ Correct Code Recall: **96%**
- 📊 Total Training Samples: **4,740**
