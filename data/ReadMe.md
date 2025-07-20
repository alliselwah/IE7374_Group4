# RecipeNLG Dataset

## Overview

**RecipeNLG** is a large-scale, structured dataset consisting of over 2.2 million cooking recipes. Developed by researchers from the Poznań University of Technology, it was designed specifically for natural language generation (NLG) tasks within the food domain. RecipeNLG extends the Recipe1M+ dataset with additional cleaning, deduplication, and semantic annotations, making it one of the most comprehensive and high-quality datasets available for research in recipe generation and food-related language modeling.

---

## Dataset Features

Each recipe in the dataset includes the following structured fields:

- **Title**: The name of the recipe.
- **Ingredients**: A semi-structured list of ingredients, including quantities, units, and ingredient names.
- **Instructions**: A sequence of step-by-step natural language directions for preparing the dish.
- **Named Entities (Optional)**: Some versions of the dataset include food-related named entities, enabling further exploration of semantic relationships between ingredients and cooking steps.

This structured format enables both supervised and unsupervised machine learning workflows for a variety of NLP tasks, including but not limited to sequence-to-sequence generation, named entity recognition, ingredient substitution, and data-to-text generation.

---

## Motivation and Use Cases

The RecipeNLG dataset addresses limitations found in previous recipe corpora (e.g., noise and inconsistency in Recipe1M+) and provides a robust benchmark for:

- **Sequence-to-sequence instruction generation**  
- **Prompt-based text generation using LLMs**  
- **Entity recognition and ingredient normalization**  
- **Similarity and recommendation systems for recipes and ingredients**  
- **Evaluation of text generation models using BLEU, ROUGE, and other metrics**

---

## Access and Format

The dataset is available for download on Kaggle:

**Kaggle URL**: [https://www.kaggle.com/datasets/saldenisov/recipenlg](https://www.kaggle.com/datasets/saldenisov/recipenlg)

**Format**: Comma-separated values (CSV)  
**Size**: ~2.2 million recipes  
**Fields**: `title`, `ingredients`, `instructions`, and optional `NER` annotations

Example usage in Python:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('recipenlg.csv')

# Display sample recipe
print(df.loc[0, 'title'])
print(df.loc[0, 'ingredients'])
print(df.loc[0, 'instructions'])'''

---

## Acknowledgements

We acknowledge the authors of RecipeNLG for their contribution to open-access research datasets and their commitment to advancing the field of data-to-text generation and natural language processing.
