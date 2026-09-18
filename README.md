
# Customer Support Triage

An NLP-based machine learning project that automatically classifies customer support tickets by **ticket type** and **priority** using text classification techniques. The project explores data preprocessing, leakage prevention, TF-IDF feature extraction, and classical machine learning models to build a reliable support ticket classification pipeline.

## Project Overview

Customer support teams receive a large number of tickets that need to be categorized and prioritized. Manual classification can be time-consuming and inconsistent.

This project aims to automate the initial triage process by predicting:

- **Ticket Type:** Incident, Problem, Request, and Change
- **Ticket Priority:** High, Medium, and Low

The project compares Logistic Regression and Linear SVM models to evaluate their effectiveness in classifying support ticket text.

## Objectives

- Understand and analyze customer support ticket data.
- Perform text preprocessing and data cleaning.
- Identify duplicate records and prevent data leakage.
- Convert textual data into numerical features using TF-IDF.
- Train and evaluate classical machine learning classification models.
- Compare model performance using accuracy, precision, recall, and F1-score.
- Analyze classification errors and identify areas for improvement.

## Dataset

The project uses the [Customer Support Tickets dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) available on Hugging Face.

The dataset contains customer support ticket information, including ticket text and classification labels.

The project focuses on the following prediction tasks:

### Ticket Type Classification

- Incident
- Problem
- Request
- Change

### Ticket Priority Classification

- High
- Medium
- Low

## Project Workflow

```text
Dataset Collection
        |
        v
Exploratory Data Analysis
        |
        v
Text Preprocessing
        |
        v
Duplicate Detection
        |
        v
Data Leakage Prevention
        |
        v
Group-Based Train/Validation/Test Split
        |
        v
TF-IDF Feature Extraction
        |
        v
Logistic Regression Baseline
        |
        v
Linear SVM Classification
        |
        v
Model Evaluation and Comparison
        |
        v
Error Analysis
```

## Data Preprocessing

The following preprocessing steps were performed:

- Analyzed the dataset structure and missing values.
- Combined the ticket subject and body into a single text field.
- Cleaned the text for feature extraction.
- Identified duplicate and repeated ticket content.
- Investigated label consistency among repeated text entries.
- Used group-based data splitting to reduce the risk of data leakage.

### Data Leakage Prevention

Repeated ticket content can cause the same or nearly identical information to appear across training and evaluation datasets. This can lead to overly optimistic performance estimates.

To reduce this risk, `GroupShuffleSplit` was used with cleaned ticket text as the grouping variable.

| Dataset | Samples |
|---|---:|
| Training | 19,768 |
| Validation | 4,226 |
| Test | 4,267 |
| **Total** | **28,261** |

The overlap checks produced the following results:

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

## Feature Extraction

Text was converted into numerical representations using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The vectorizer configuration was:

```python
TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
```

### Configuration

- `max_features=20000`: Limits the feature vocabulary to 20,000 features.
- `ngram_range=(1, 2)`: Uses unigrams and bigrams.
- `min_df=2`: Excludes terms appearing in fewer than two documents.
- `max_df=0.95`: Excludes terms appearing in more than 95% of documents.

The vectorizer was fitted only on the training data and then used to transform the validation and test sets.

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression was used as the baseline classification model.

It provides a simple and efficient approach for text classification using high-dimensional TF-IDF features.

### 2. Linear SVM

Linear Support Vector Machine was evaluated as an alternative classical machine learning model.

Linear SVM is suitable for high-dimensional, sparse text representations and was used to determine whether it could improve upon the Logistic Regression baseline.

## Model Results

The models were evaluated on the held-out test set.

### Ticket Type Classification

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 84.39% | 0.84 | 0.84 |
| Linear SVM | **86.43%** | **0.87** | **0.86** |

Linear SVM improved test accuracy by **2.04 percentage points** compared with Logistic Regression.

The largest class-level improvement was observed for the **Problem** class, where F1-score increased from 0.59 to 0.66.

### Ticket Priority Classification

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 61.28% | 0.56 | 0.60 |
| Linear SVM | **64.66%** | **0.62** | **0.64** |

Linear SVM improved test accuracy by **3.38 percentage points** compared with Logistic Regression.

The largest improvement was observed for the **Low priority** class:

| Metric | Logistic Regression | Linear SVM |
|---|---:|---:|
| F1-score | 0.38 | **0.52** |

## Error Analysis

Model evaluation showed that ticket priority classification was more challenging than ticket type classification.

The confusion matrix and error analysis revealed that some tickets contained terms associated with security, sensitive information, healthcare, or compliance but were not necessarily urgent.

This highlighted an important limitation of lexical text features:

> Words that sound important do not always indicate the actual urgency or business impact of a ticket.

Although Linear SVM improved performance, priority classification remains challenging because ticket priority can depend on context, business impact, and urgency that may not be explicitly expressed in the text.

## Key Learnings

- Data quality and evaluation design are important before model training.
- Duplicate records can create data leakage and misleading evaluation results.
- TF-IDF provides an effective baseline for converting text into numerical features.
- Logistic Regression and Linear SVM are practical approaches for classical text classification.
- Accuracy alone does not fully describe model performance.
- Class-level F1-score and confusion matrices help identify model weaknesses.
- Error analysis provides insights that cannot be obtained from a single performance metric.
- Better classification performance does not necessarily mean that the underlying problem has been completely solved.

## Project Structure

```text
customer-support-triage/
│
├── 01_dataset_understanding_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_model_training.ipynb
├── 04_model_comparison.ipynb
│
├── train.csv
├── validation.csv
├── test.csv
│
├── .gitignore
└── README.md
```

> Note: Dataset CSV files are excluded from version control where applicable.

## Technologies and Tools

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** TF-IDF
- **Models:** Logistic Regression, Linear SVM
- **Data Visualization:** Matplotlib
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git, GitHub

## Future Improvements

The next stages of the project include:

- Experimenting with Transformer-based models such as DistilBERT.
- Comparing contextual text representations with TF-IDF features.
- Performing additional model evaluation and error analysis.
- Exploring model saving and inference workflows.
- Developing an API for ticket classification.
- Building an interactive interface for support ticket predictions.
- Evaluating model performance in a practical deployment scenario.

## Conclusion

This project demonstrates an end-to-end classical NLP classification workflow, from dataset exploration and leakage prevention to feature extraction, model training, and evaluation.

The comparison between Logistic Regression and Linear SVM showed that Linear SVM achieved better performance on both ticket type and priority classification. The results also demonstrated the importance of class-level evaluation and error analysis when working with real-world text classification problems.

The project will be extended by exploring Transformer-based models to investigate whether contextual language representations can improve classification performance.

## Author

**Jasmine Sardana**

[GitHub](https://github.com/Jasmine-Sardana)
