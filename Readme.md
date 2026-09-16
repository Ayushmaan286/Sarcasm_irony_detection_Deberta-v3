# Context-Aware Joint Sarcasm and Irony Detection using DeBERTa-v3

A transformer-based NLP system for detecting sarcasm and irony using a shared DeBERTa-v3-base encoder with separate classification heads for each task. The system incorporates conversational context when available and provides predictions through a Streamlit web application.

---

##  PROJECT OVERVIEW

Sarcasm and irony are challenging Natural Language Processing (NLP) tasks because the intended meaning of an utterance may differ from its literal meaning. In conversational settings, surrounding context can provide additional information needed to interpret the current utterance.

This project develops a **joint sarcasm and irony detection system using DeBERTa-v3-base**.

The system combines:

- Context-aware input construction
- A shared DeBERTa-v3-base encoder
- Separate sarcasm and irony classification heads
- Masked multi-task learning
- Experimental analysis and error analysis
- A standalone inference pipeline
- A Streamlit-based web interface

---

## OBJECTIVES

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

# DATASET

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

The resulting dataset contains:

id
context
text
sarcasm_label
irony_label
source


# LABEL REPRESENTATION

0 -->Negative
1 -->Positive
-1 -->Annotation unavailable 