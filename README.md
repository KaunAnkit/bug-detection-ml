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

## Features

- **Automated Data Processing**: Convert raw code files into ML-ready format
- **TF-IDF Vectorization**: Extract meaningful features from code syntax
- **Model Persistence**: Save and load trained models for reuse
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, and recall
- **Simple Prediction API**: Easy integration into existing workflows

## Installation

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
│   ├── evaluation_report.txt    # Model performance metrics
│   ├── trained_model.pkl        # Trained Logistic Regression classifier
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
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## 📖 Usage Guide

### Complete Pipeline Execution

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
cat models/evaluation_report.txt
```

The report contains:
- Overall accuracy score
- Confusion matrix
- Detailed classification report (precision, recall, F1-score for each class)

### Generated Model Files

The pipeline creates the following files in the `models/` directory:

- `vectorizer.pkl` - TF-IDF vectorizer (saved by feature_extraction.py)
- `X_Tfidf.pkl` - Transformed feature matrix (saved by feature_extraction.py)
- `y.pkl` - Target labels (saved by feature_extraction.py)
- `trained_model.pkl` - Trained Logistic Regression model (saved by model_training.py)
- `evaluation_report.txt` - Performance metrics (saved by model_evaluation.py)

## 📊 Model Performance

Current model metrics on the test set:

| Metric | Value |
|--------|-------|
| **Accuracy** | 55.56% |
| **Precision (Buggy Code)** | 0.62 |
| **Recall (Buggy Code)** | 0.83 |
| **F1-Score (Buggy Code)** | 0.71 |

### Confusion Matrix

```
                Predicted
              Buggy  Correct
Actual Buggy   [5]     [1]
     Correct   [3]     [0]
```

**Note**: The model shows higher recall (83%) than precision (62%), meaning it's better at catching bugs but may produce false positives (flagging correct code as buggy).

## 💾 Dataset

### Data Source

The training dataset is **AI-generate** due to the scarcity of publicly available, labeled bug datasets. 

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

##  Technical Details

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
- Parameters: `max_iter=1000, random_state=42`

### Training Process

1. **Data Preprocessing** (`data_preprocessing.py`):
   - Reads code files from `data/raw/bug/` and `data/raw/correct/`
   - Creates labeled dataset with columns: Code_text, label
   - Saves to `data/processed/cleaned_data.csv`

2. **Feature Extraction** (`feature_extraction.py`):
   - Loads cleaned data
   - Applies TF-IDF vectorization to code text
   - Saves vectorizer and transformed features as pickle files

3. **Model Training** (`model_training.py`):
   - Loads TF-IDF features and labels
   - Splits data (80% train, 20% test, random_state=42)
   - Trains Logistic Regression classifier (max_iter=1000)
   - Saves trained model

4. **Model Evaluation** (`model_evaluation.py`):
   - Loads trained model and test data
   - Generates predictions
   - Calculates metrics and saves evaluation report

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Limited Training Data**: AI-generated dataset may not reflect real-world complexity
2. **Low Accuracy**: 55% accuracy indicates room for improvement
3. **Syntax-Only Analysis**: Doesn't understand code semantics or execution flow
4. **Language-Specific**: Currently optimized for Python code only
5. **Simple Model**: Logistic Regression may not capture complex bug patterns

### Planned Improvements

- [ ] **Collect Real-World Data**: Integrate with bug tracking systems (JIRA, GitHub Issues)
- [ ] **Advanced Models**: Experiment with Random Forest, XGBoost, or Neural Networks
- [ ] **Code-Specific Features**: 
  - Abstract Syntax Tree (AST) analysis
  - Cyclomatic complexity
  - Code metrics (lines, depth, dependencies)
- [ ] **Deep Learning**: Implement CodeBERT or GraphCodeBERT for semantic understanding
- [ ] **Multi-Language Support**: Extend to Java, C++, JavaScript
- [ ] **Explainability**: Add SHAP or LIME for model interpretability
- [ ] **Active Learning**: Improve model with user feedback
- [ ] **IDE Integration**: Create plugins for VS Code, PyCharm

## Contributing

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Ankit Jha - [LinkedIn](www.linkedin.com/in/connectwithankitjha) - connectwithankitjha@gmail.com

Project Link: [https://github.com/KaunAnkit/bug-detection-ml](https://github.com/KaunAnkit/bug-detection-ml)

## Acknowledgments

- Inspired by research in automated program repair and static analysis
- TF-IDF implementation from scikit-learn
- Code examples generated with assistance from AI tools

---


**⚠️ Disclaimer**: This is a proof-of-concept project. The model should not be used as the sole method for bug detection in production systems. Always combine automated tools with manual code review and comprehensive testing.
