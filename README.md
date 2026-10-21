# Arabic Sentiment Analyzer

An Arabic Natural Language Processing (NLP) project that classifies social media text into **Positive, Neutral, and Negative** sentiment using traditional machine learning and a fine-tuned MARBERT transformer.

The project covers data preprocessing, quality analysis, model training, evaluation, and an interactive Gradio demo.

## Overview

Arabic sentiment analysis is challenging due to dialectal variation, informal expressions, contextual meanings, and noisy social media text.

In this project, I developed and compared two approaches:

1. **Baseline:** TF-IDF with Logistic Regression.
2. **Transformer:** Fine-tuned MARBERT, a pretrained Arabic language model.

Both models were evaluated on the same held-out test set to compare their classification performance.

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Data Processing:** Pandas, NumPy, Hugging Face Datasets
- **Evaluation:** Accuracy, Precision, Recall, Macro-F1, Confusion Matrices
- **Interface:** Gradio
- **Development Environment:** Google Colab

## Dataset

**Dataset:** [ArSAS — Arabic Sentiment Analysis Dataset](https://huggingface.co/datasets/arbml/ArSAS)

### Data Preparation

I performed the following preprocessing steps:

- Selected the relevant text and sentiment columns.
- Excluded the Mixed category to create a three-class classification task.
- Removed missing, empty, and duplicate text records.
- Applied basic text cleaning to remove URLs, mentions, and unnecessary whitespace.
- Investigated suspicious entries containing annotation metadata and concatenated tweets.
- Excluded 70 potentially corrupted records.

After initial preprocessing, the dataset contained 18,511 records. Following the quality audit, the final dataset contained **18,441 examples**.

I divided the cleaned dataset into:

- **80% Training**
- **10% Validation**
- **10% Testing**

The same finalized splits were used for both models.


## Methodology

### 1. Baseline: TF-IDF + Logistic Regression

I developed a traditional machine learning baseline using TF-IDF features with unigrams and bigrams, followed by Logistic Regression.

This provided a benchmark for evaluating the performance of a more advanced transformer-based approach.

### 2. MARBERT Fine-Tuning

I fine-tuned the pretrained `UBC-NLP/MARBERT` model for three-class Arabic sentiment classification.

The workflow included:

- Tokenizing Arabic text using MARBERT's pretrained tokenizer.
- Running a pilot training experiment.
- Fine-tuning the model on the full training dataset.
- Evaluating performance using accuracy and macro-F1.
- Selecting the best checkpoint based on validation macro-F1.
- Conducting final evaluation on the held-out test set.

## Results

Both models were evaluated on the same **1,845 previously unseen test examples**.

| Metric | Logistic Regression | MARBERT |
|---|---:|---:|
| Test Accuracy | 77.02% | **83.58%** |
| Test Macro-F1 | 0.7541 | **0.8248** |

MARBERT improved test accuracy by **6.56 percentage points** and macro-F1 by **0.0707** compared with the baseline.

### Confusion Matrices

**Baseline — Validation Set**

<img src="images%3A/baseline_validation_confusion_matrix.png" width="500" alt="Baseline Confusion Matrix">

**MARBERT — Test Set**

<img src="images%3A/marbert_test_confusion_matrix.png" width="500" alt="MARBERT Confusion Matrix">

## Interactive Gradio Demo

I built an interactive Gradio application that accepts Arabic text and predicts its sentiment using my fine-tuned MARBERT model.

The interface displays the predicted sentiment alongside probability scores for all three categories.

### Positive Sentiment

![Positive Prediction](images%3A/demo_positive.png)

### Negative Sentiment

![Negative Prediction](images%3A/demo_negative.png)

### Neutral Sentiment

![Neutral Prediction](images%3A/demo_neutral.png)

**Deployment status:** The demo has been tested locally in Google Colab. Permanent deployment is planned.

## Error Analysis

During evaluation, I identified several classification challenges:

- **Dialectal Arabic:** Informal expressions may convey meanings that differ from their literal interpretations.
- **Context-dependent sentiment:** The same word can have different sentiment implications depending on context.
- **Class imbalance:** Positive sentiment was the most challenging category.
- **Noisy social media text:** Hashtags, emojis, and mixed-language content can complicate classification.
- **Dataset quality:** Some examples contained apparent annotation metadata and concatenated tweets.

For example, the sentence:

> الصراحة التجربة كانت رهيبة

was incorrectly classified as Negative by the baseline model but correctly classified as Positive by MARBERT.

This example illustrates a potential advantage of contextual language models for certain informal Arabic expressions, although broader dialect performance requires additional evaluation.


## Project Structure

```text
arabic-sentiment-analyzer/
│
├── notebooks/
│   ├── 01_Arabic_Sentiment_Baseline.ipynb
│   └── 02_MARBERT_Fine_Tuning.ipynb
│
├── images/
│   ├── baseline_validation_confusion_matrix.png
│   ├── marbert_test_confusion_matrix.png
│   ├── demo_positive.png
│   ├── demo_negative.png
│   └── demo_neutral.png
│
├── README.md
├── .gitignore
└── LICENSE
```

## Reproducibility

The Jupyter notebooks document the preprocessing, training, evaluation, and demonstration workflow.

1. Open `01_Arabic_Sentiment_Baseline.ipynb` in Google Colab.
2. Install the required Python libraries and load the ArSAS dataset.
3. Run the preprocessing and baseline training cells.
4. Save the finalized training, validation, and test splits.
5. Open `02_MARBERT_Fine_Tuning.ipynb`.
6. Load the saved dataset splits and run the MARBERT fine-tuning experiment using a GPU-enabled runtime.
7. Evaluate the model and launch the Gradio demo.

**Note:** Fine-tuned model weights are not included in this repository. Reproducing the complete training workflow requires downloading the pretrained MARBERT model and running the notebooks.

## Limitations

- The cleaned dataset has not been fully manually verified.
- Performance was measured on ArSAS and may differ on other Arabic datasets.
- Positive sentiment remains the most challenging category.
- The model's output probabilities should not be interpreted as perfectly calibrated confidence estimates.
- The current interactive demo depends on a saved model and a compatible Python environment.

## Future Improvements

- Evaluate the classifier on additional Arabic dialect datasets.
- Explore alternative Arabic transformer architectures.
- Investigate misclassified Positive examples.
- Improve robustness to sarcasm, negation, and mixed-language text.
- Deploy the interactive demo permanently using Hugging Face Spaces.

## License

This project is licensed under the MIT License. External datasets and pretrained models remain subject to their respective licenses.
