# News Classification System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-TF--IDF_%26_Machine_Learning-F7931E?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge)

<p align="center">
  <b>An end-to-end Natural Language Processing (NLP) multi-class news categorization pipeline designed to process raw news text, extract TF-IDF features, train machine learning classifiers, evaluate multi-class performance metrics, and perform real-time predictions.</b>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Algorithm-TF--IDF_Vectorization-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-Multi--Class_Classifier-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Handling-Class_Weight_Balanced-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Metrics-F1_Score_%26_Confusion_Matrix-red?style=for-the-badge" />
</p>

---

## Overview

**News Classification System** is an NLP-driven multi-class topic categorization pipeline built to analyze unstructured news articles and headlines, automatically categorizing them into actionable news categories (such as World, Sports, Business, Sci/Tech, etc.).

Instead of relying on simple keyword lookups, the system establishes a robust end-to-end processing and model pipeline incorporating:

* **Text Preprocessing & Normalization:** Noise removal, tokenization, stop-word filtering, and text standardization.
* **Feature Extraction:** Sublinear TF-IDF N-gram Vectorization for capturing contextual n-gram associations.
* **Class Imbalance Mitigation:** Automated inverse-frequency class weighting to handle uneven category distributions.
* **Evaluation & Benchmarking:** Detailed precision, recall, F1-score, and confusion matrix evaluations across all target classes.

The system transforms raw unstructured text into high-dimensional numerical feature vectors using TF-IDF and outputs high-accuracy topic predictions via modular execution scripts.

---

### Application Features

* **Modular End-to-End Architecture:** Clean separation of dataset loading, cleaning pipelines, feature extraction, model training, evaluation, and inference modules.
* **Multi-Class Topic Classification:** Capable of handling complex multi-class textual boundaries with optimized decision thresholds.
* **Robust Text Normalization:** Preserves critical contextual signals while stripping unwanted HTML markup, special characters, and uninformative noise.
* **Transparent Multi-Class Metrics:** Generates detailed macro/weighted F1-scores alongside full confusion matrices for error diagnosis.
* **Production-Ready Artifacts:** Automatically serializes trained models and vectorizer instances (`.pkl`) for immediate real-time prediction workflows.

---

## Project Objective

The primary objective is to build a scalable, production-ready news classification system that can:

* Clean, normalize, and structure unstructured textual news feeds from raw datasets.
* Address data quality issues like duplicate records, missing attributes, and uninformative stop-words.
* Vectorize article body text and headlines into sparse numerical representations using `TfidfVectorizer`.
* Support multi-class performance benchmarking across diverse news domains.
* Evaluate predictions using standard classification metrics (Precision, Recall, F1-Score, and Confusion Matrix).
* Maintain a clean directory structure adhering to modern Python and Machine Learning repository standards.

---

## Problem Statement

Categorizing massive streams of daily news articles manually is inefficient and slow, while simple keyword searches fail to capture contextual semantics—leading to inaccurate content recommendation and indexing.

Standard baseline solutions frequently suffer from:

* Difficulty distinguishing subtle topic overlaps between adjacent categories (e.g., Business vs. Technology).
* Loss of important contextual semantics due to aggressive or improper text preprocessing routines.
* Unstructured script organizations that lack modularity, clean entry points, and deployment-ready model serialization.

### Proposed Solution

This project introduces a robust NLP classification pipeline executing:

$$\text{Raw News Text} \longrightarrow \text{Text Preprocessing} \longrightarrow \text{TF-IDF Matrix} \longrightarrow \text{Model Training} \longrightarrow \text{Topic Prediction}$$

For every news item processed, the system produces:

```text
Cleaned News Text
Predicted Topic Category
Class Confidence / Decision Scores
Evaluation Benchmarks & Confusion Matrix Breakdown
