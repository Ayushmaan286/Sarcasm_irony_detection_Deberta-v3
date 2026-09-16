# Context-Aware Joint Sarcasm and Irony Detection using DeBERTa-v3

A transformer-based Natural Language Processing (NLP) system for detecting sarcasm and irony using a shared DeBERTa-v3-base encoder with separate classification heads for each task. The system incorporates conversational context when available and provides predictions through a Streamlit web application.

---

## 📌 Project Overview

Sarcasm and irony are challenging Natural Language Processing tasks because the intended meaning of an utterance may differ from its literal meaning. In conversational settings, surrounding context can provide additional information for interpreting the current utterance.

This project develops a **joint sarcasm and irony detection system using DeBERTa-v3-base**.

The system combines:

- Context-aware input construction
- A shared DeBERTa-v3-base transformer encoder
- Separate sarcasm and irony classification heads
- Masked multi-task learning
- Experimental and error analysis
- A standalone inference pipeline
- A Streamlit-based web interface

---

##  Objectives

The main objectives of this project are:

- Develop a transformer-based sarcasm detection system.
- Develop an irony detection system within the same architecture.
- Incorporate conversational context when available.
- Jointly train sarcasm and irony classification tasks.
- Handle datasets with partially available task annotations.
- Analyze the effect of conversational context.
- Perform error analysis and hyperparameter experiments.
- Deploy the trained model through a web-based interface.

---

#  Dataset

Two publicly available datasets were integrated into a unified dataset.

## 1. Self-Annotated Reddit Corpus (SARC)

SARC is a Reddit-based dataset containing sarcastic utterances along with conversational context.

After preprocessing:

- **19,920 examples**
- Task: Sarcasm Detection
- Conversational context: Available

## 2. SemEval-2018 Task 3

SemEval-2018 Task 3 provides Twitter data annotated for irony.

The training data used in this project contains:

- **3,817 examples**
- Task: Irony Detection
- Conversational context: Not available

## Unified Dataset

The two datasets were combined into a unified dataset containing:

**23,737 examples**

The resulting dataset contains the following columns:


id
context
text
sarcasm_label
irony_label
source


### Label Representation


0  → Negative class
1  → Positive class
-1 → Annotation unavailable


The value `-1` is not treated as a third classification class. It indicates that the corresponding task is not annotated for that example.

---

#  Dataset Split

The unified dataset was divided into training and validation sets using an approximately 80/20 split.

| Split | Examples |
|---|---:|
| Training | 18,989 |
| Validation | 4,748 |
| Total | 23,737 |

Source- and label-aware stratification was used during the split.

---

# Methodology

The proposed system uses a shared transformer encoder followed by two independent classification heads.

```text
                  Context + Current Text
                           │
                           ▼
                  DeBERTa-v3-base
                           │
                           ▼
                 Shared Representation
                        (768)
                           │
                        Dropout
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
          Sarcasm Head          Irony Head
            768 → 2               768 → 2
                 │                   │
                 ▼                   ▼
          Sarcasm Output        Irony Output
```

The architecture contains:

- One shared DeBERTa-v3-base encoder
- One sarcasm classification head
- One irony classification head
- A shared dropout layer

The two tasks therefore use a common textual representation while producing independent predictions.

---

#  Context-Aware Input

When conversational context is available, the model receives both the context and the current utterance.

The input is constructed as:

```text
[CONTEXT]
{context}

[CURRENT]
{text}
```

For examples where context is unavailable:

```text
[CONTEXT]
No context available.

[CURRENT]
{text}
```

This provides a consistent input format for both datasets.

---

#  Masked Multi-Task Learning

The two datasets do not contain annotations for both tasks.

For example:

- SARC contains sarcasm labels but does not provide irony annotations.
- SemEval-2018 Task 3 contains irony labels but does not provide sarcasm annotations.

To handle this, unavailable labels are represented using `-1`.

The loss is calculated only for tasks with valid annotations.

Conceptually:

```text
Sarcasm Loss = Cross Entropy(valid sarcasm labels)

Irony Loss = Cross Entropy(valid irony labels)

Total Loss = Sarcasm Loss + Irony Loss
```

Therefore, an example without an irony annotation contributes to the sarcasm loss but not to the irony loss, and vice versa.

---

#  Model Configuration

The project uses:

```text
Model: microsoft/deberta-v3-base
```

### Model Details

| Parameter | Value |
|---|---|
| Base Model | DeBERTa-v3-base |
| Transformer Layers | 12 |
| Hidden Size | 768 |
| Classification Heads | 2 |
| Classes per Task | 2 |
| Dropout | 0.1 |
| Maximum Sequence Length | 256 |
| Approx. Parameters | 183.8 million |

---

#  Training Configuration

The baseline model was trained using Google Colab with an NVIDIA Tesla T4 GPU.

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 4 |
| Epochs | 3 |
| Maximum Sequence Length | 256 |
| Dropout | 0.1 |
| Loss Weighting | 1.0 : 1.0 |
| Hardware | NVIDIA Tesla T4 |

---

# Results

The baseline model was evaluated on the validation set.

## Sarcasm Detection

| Metric | Score |
|---|---:|
| Accuracy | **71.84%** |
| Precision | **69.47%** |
| Recall | **78.23%** |
| F1-score | **73.59%** |

## Irony Detection

| Metric | Score |
|---|---:|
| Accuracy | **76.67%** |
| Precision | **73.60%** |
| Recall | **82.89%** |
| F1-score | **77.97%** |

### Combined F1

The average of the sarcasm and irony F1-scores is:

**75.78%**

The best checkpoint was selected using the combined validation F1.

---

#  Context Ablation Study

A context-versus-text-only experiment was performed on the SARC validation subset.

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Context + Text | 71.87% | 69.50% | 78.23% | **73.60%** |
| Text Only | 69.41% | 66.14% | 79.88% | 72.36% |
| Difference | +2.46 pts | +3.36 pts | -1.65 pts | **+1.24 pts** |

The context-aware configuration produced a **1.24 percentage-point increase in sarcasm F1** on this evaluation.

The primary observed change was improved precision accompanied by a small decrease in recall.

---

#  Experimental Analysis

Several controlled experiments were conducted to investigate the behavior of the model.

Experiments included:

- Learning-rate variation
- Number of training epochs
- Multi-task loss weighting
- Maximum sequence length
- Context-preserving truncation
- Context versus text-only input

Selected experiment results are shown below.

| Experiment | Combined F1 |
|---|---:|
| Max Length = 128 | 71.23% |
| Max Length = 256 | 73.49% |
| Max Length = 384 | 74.92% |
| Loss Weights = 2:1 | 73.93% |
| Loss Weights = 1:2 | 74.71% |
| Epochs = 2 | 71.52% |
| Epochs = 4 | 71.25% |

The controlled experiments did not exceed the original baseline result of **75.78% combined F1**.

Therefore, the original baseline configuration and checkpoint were retained.

---

#  Error Analysis

Error analysis was performed for both sarcasm and irony classification.

## Sarcasm

The SARC validation subset contained:

```text
True Negatives:  1301
False Positives: 686
False Negatives: 435
True Positives:  1563
```

False-negative utterances tended to be shorter than the other error categories.

Frequently occurring terms in some false-positive examples included:

```text
yeah
well
sure
should
because
why
```

Frequently occurring terms in some false-negative examples included:

```text
mean
point
shit
right
though
```

These observations highlight short and ambiguous utterances as areas for further investigation.

## Irony

The irony validation subset contained:

```text
True Negatives: 270
False Positives: 113
False Negatives: 65
True Positives: 315
```

The analysis found URL-related patterns such as `http` among some error cases.

Other frequently occurring terms included:

```text
christmas
happy
day
today
family
```

These observations provide directions for further analysis of social-media-specific patterns and contextual interpretation.

---

#  Confusion Matrices

## Sarcasm

```text
                 Predicted
              Negative Positive

Actual
Negative        1301     686
Positive         435    1563
```

## Irony

```text
                 Predicted
              Negative Positive

Actual
Negative         270     113
Positive          65     315
```

The confusion matrices provide a detailed view of false-positive and false-negative behavior for both tasks.

---

#  Training Analysis

Training and validation curves were generated for:

- Training loss
- Validation loss
- Sarcasm F1
- Irony F1
- Combined F1

Validation F1 values across the three epochs were:

| Epoch | Sarcasm F1 | Irony F1 | Combined F1 |
|---|---:|---:|---:|
| 1 | 70.68% | 73.64% | 72.16% |
| 2 | 70.50% | 76.01% | 73.26% |
| 3 | **73.59%** | **77.97%** | **75.78%** |

The third epoch produced the highest combined validation F1.

---

#  Truncation Analysis

The baseline model uses:

```text
Maximum sequence length = 256 tokens
```

An investigation showed that long context-plus-text sequences can be truncated.

A context-preserving truncation strategy was implemented and validated. However, controlled retraining using the modified strategy produced a slightly lower validation result than the original baseline.

Therefore, the original baseline preprocessing strategy was retained.

The 256-token sequence limit remains a known limitation of the current implementation.

---

#  Standalone Inference

The standalone prediction pipeline is implemented in:

```text
inference/inference.py
```

The pipeline performs:

```text
Context + Current Text
        │
        ▼
Input Construction
        │
        ▼
DeBERTa-v3 Tokenization
        │
        ▼
256-token Truncation
        │
        ▼
DeBERTa-v3-base
        │
        ├───────────────┐
        ▼               ▼
   Sarcasm Head     Irony Head
        │               │
        ▼               ▼
     Softmax         Softmax
        │               │
        ▼               ▼
   Prediction       Prediction
```

The `predict()` function returns:

- Predicted label
- Confidence
- Probability for each class

---

#  Streamlit Application

The trained model is deployed through a Streamlit web application.

The application provides:

- Optional conversational context
- Required current text
- Sarcasm classification
- Irony classification
- Confidence values
- Class probabilities
- Input validation
- Prediction error handling

The Streamlit application is located at:

```text
streamlit_app/app.py
```

---

#  Running the Application Locally

## Requirements

The application uses:

- Python 3.11
- PyTorch
- Transformers
- SentencePiece
- Streamlit

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate on Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

For CPU-based installation:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece streamlit
```

### Run Streamlit

From the project directory:

```bash
cd streamlit_app
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

The trained checkpoint must be available locally at:

```text
models/best_checkpoint.pt
```

The checkpoint is intentionally excluded from the Git repository because of its large file size.

---

#  Project Structure

```text
sarcasm_irony_project/
│
├── data/
│
├── inference/
│   └── inference.py
│
├── models/
│   └── best_checkpoint.pt
│
├── results/
│
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   └── RUN_INSTRUCTIONS.txt
│
├── .gitignore
└── README.md
```

The `.venv/` directory is used only for the local Python environment and is excluded from version control.

The trained checkpoint is also excluded from GitHub because of its large file size.

---

#  Deployment Validation

The final local deployment was tested using seven validation tests.

| Test | Result |
|---|---|
| Normal contextual prediction | ✅ PASS |
| No-context prediction | ✅ PASS |
| Empty-text validation | ✅ PASS |
| Short text | ✅ PASS |
| Long input / truncation | ✅ PASS |
| Streamlit vs direct inference consistency | ✅ PASS |
| Checkpoint integrity | ✅ PASS |

The local checkpoint was also verified using a SHA-256 integrity check.

---

#  Limitations

The current system has several limitations:

1. The model uses a maximum sequence length of 256 tokens.
2. Long context-plus-text inputs can therefore be truncated.
3. SARC provides conversational context while SemEval-2018 Task 3 does not.
4. The two tasks are not jointly annotated for every example.
5. Very short or ambiguous utterances can be difficult to classify.
6. Evaluation is based on the available validation split.
7. Confidence values represent model probabilities and should not be interpreted as guaranteed certainty.

---

#  Future Work

Potential future improvements include:

- More sophisticated context selection
- Improved handling of long conversational threads
- Additional conversational datasets
- Probability calibration
- More extensive hyperparameter optimization
- Comparison with additional transformer architectures
- Larger-scale evaluation
- Improved handling of URLs and social-media artifacts
- Human evaluation of difficult examples
- Further investigation of joint sarcasm-irony representations

---

#  Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **DeBERTa-v3**
- **SentencePiece**
- **Streamlit**
- **Google Colab**
- **NVIDIA Tesla T4**

---

#  Project Status

**Working prototype with validated local deployment**

The project currently includes:

- A trained DeBERTa-v3-based joint sarcasm and irony classifier
- Context-aware preprocessing
- Masked multi-task learning
- Experimental and error analysis
- Standalone inference
- Streamlit deployment
- End-to-end local validation

The system provides a foundation for further research and experimentation in context-aware sarcasm and irony detection.