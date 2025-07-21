# Recipe Generator using Seq2Seq Model

This project implements a Sequence-to-Sequence (Seq2Seq) model for generating coherent cooking recipe instructions from a list of ingredients. It was developed as part of coursework for the IE7374 - Generative AI course.

## Overview

Given the increasing use of generative models in natural language processing, this project explores how a Seq2Seq architecture based on LSTM layers can be used to learn the mapping from ingredients to cooking instructions using the RecipeNLG dataset.

The notebook demonstrates a complete machine learning pipeline:
- Data loading and preprocessing
- Tokenization and sequence padding
- Model architecture and training
- Recipe instruction generation
- Evaluation and qualitative analysis

## Dataset

- **Source**: [RecipeNLG - Kaggle](https://www.kaggle.com/datasets/saldenisov/recipenlg)
- **Format**: JSON
- **Fields used**: `ingredients` and `instructions`
- **Preprocessing steps**:
  - Extracted relevant fields
  - Lowercased and cleaned the text
  - Removed recipes with missing or overly short content
  - Tokenized and padded sequences to uniform lengths

## Model Architecture

The model uses a Seq2Seq architecture built with LSTM layers:
- **Encoder**: Embedding layer followed by an LSTM to encode the ingredients
- **Decoder**: Embedding layer, LSTM, and Dense layer to generate instructions
- **Loss function**: Sparse categorical crossentropy
- **Optimizer**: Adam
- **Training**: 10 epochs using teacher forcing

## Evaluation

The model is evaluated using:
- Training and validation loss
- BLEU score
- Manual inspection of generated recipe instructions

Several sample generations are included to illustrate the model’s ability to produce multi-step instructions that are syntactically fluent and semantically relevant.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Kaggle API (for dataset access)

Dependencies can be installed using the following command:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib kaggle
```
