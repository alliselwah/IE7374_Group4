# Generative AI Recipe Creation Tool

## Project Title & Description
This project introduces a generative AI-powered tool designed to assist users in creating meal ideas from ingredients they already have at home. By simply inputting a list of grocery items, users receive one or more recipe suggestions that effectively incorporate those items. The system leverages natural language processing and advanced generative models to craft complete and coherent recipes, including titles, ingredients lists, step-by-step instructions, and potential preparation times. This tool aims to reduce food waste, save time, and inspire home cooking through the innovative application of artificial intelligence.

## Final Topic Area
Natural Language Processing (NLP), with a specific focus on Controlled Text Generation. The project's core involves generating simple, easy-to-make recipes based on natural language ingredient inputs. This process emphasizes sequence generation and semantic understanding, applying various NLP techniques such as tokenization, embedding, and leveraging transformer-based models for coherent and contextually relevant recipe outputs.

## Dataset Description
For this project, we primarily utilize the **RecipeNLG Dataset**, publicly available on Kaggle. This extensive dataset comprises approximately two million recipes, each richly detailed with a title, a list of ingredients (extracted as Named Entities, 'NER'), and comprehensive cooking instructions. Our focus is exclusively on these textual components, which are crucial for training generative language models. The data is provided in a convenient CSV format, facilitating straightforward parsing and manipulation within our data pipeline.

**DOWNLOAD LINK DUE TO FILE BEING TOO LARGE:** [https://www.kaggle.com/datasets/saldenisov/recipenlg](https://www.kaggle.com/datasets/saldenisov/recipenlg)

To get started, you must download `full_dataset.csv` from the RecipeNLG Kaggle page and place it into the `project-root/data/raw/` directory.

## Model Selection
We leverage state-of-the-art transformer-based generative language models from the Hugging Face Transformers library. Our selected models include:

* GPT-2 (both PyTorch and TensorFlow implementations)
* BART (PyTorch implementation for conditional generation)

These models are chosen for their proven capabilities in natural language generation and their ability to produce coherent, contextually relevant text. We use pre-trained versions as a base and fine-tune them on the RecipeNLG dataset to adapt their generation capabilities specifically to the cooking domain.

## Research Questions
* How effective is a fine-tuned language model (e.g., GPT-2, BART) in generating complete, coherent, and practical recipes from a diverse list of grocery items?
* To what extent can transformer-based models generalize to novel ingredient combinations or recipe types not explicitly encountered during their training on the RecipeNLG dataset?
* How do different prompt structures (e.g., including cooking context, cuisine hints, or just raw ingredient lists) influence the quality, specificity, and creativity of the generated recipes?
* (Planned Future Work) Does incorporating advanced techniques like ingredient embeddings or clustering (e.g., to understand "butter vs. margarine") improve semantic matching, ingredient substitution logic, and overall recipe quality?

## Project Structure

Okay, I understand. You want to update your README.md file with the specific content you've provided, which is essentially the "Project Title & Description" through "Model Usage and Reproduction" sections, but formatted as proper Markdown.

This content is a mix of headings and paragraphs, some of which you previously had, but now needs to be combined and formatted correctly.

Here's the Markdown content for your README.md based on your latest input:

Markdown

# Generative AI Recipe Creation Tool

## Project Title & Description
This project introduces a generative AI-powered tool designed to assist users in creating meal ideas from ingredients they already have at home. By simply inputting a list of grocery items, users receive one or more recipe suggestions that effectively incorporate those items. The system leverages natural language processing and advanced generative models to craft complete and coherent recipes, including titles, ingredients lists, step-by-step instructions, and potential preparation times. This tool aims to reduce food waste, save time, and inspire home cooking through the innovative application of artificial intelligence.

## Final Topic Area
Natural Language Processing (NLP), with a specific focus on Controlled Text Generation. The project's core involves generating simple, easy-to-make recipes based on natural language ingredient inputs. This process emphasizes sequence generation and semantic understanding, applying various NLP techniques such as tokenization, embedding, and leveraging transformer-based models for coherent and contextually relevant recipe outputs.

## Dataset Description
For this project, we primarily utilize the **RecipeNLG Dataset**, publicly available on Kaggle. This extensive dataset comprises approximately two million recipes, each richly detailed with a title, a list of ingredients (extracted as Named Entities, 'NER'), and comprehensive cooking instructions. Our focus is exclusively on these textual components, which are crucial for training generative language models. The data is provided in a convenient CSV format, facilitating straightforward parsing and manipulation within our data pipeline.

**DOWNLOAD LINK DUE TO FILE BEING TOO LARGE:** [https://www.kaggle.com/datasets/saldenisov/recipenlg](https://www.kaggle.com/datasets/saldenisov/recipenlg)

To get started, you must download `full_dataset.csv` from the RecipeNLG Kaggle page and place it into the `project-root/data/raw/` directory.

## Model Selection
We leverage state-of-the-art transformer-based generative language models from the Hugging Face Transformers library. Our selected models include:

* GPT-2 (both PyTorch and TensorFlow implementations)
* BART (PyTorch implementation for conditional generation)

These models are chosen for their proven capabilities in natural language generation and their ability to produce coherent, contextually relevant text. We use pre-trained versions as a base and fine-tune them on the RecipeNLG dataset to adapt their generation capabilities specifically to the cooking domain.

## Research Questions
* How effective is a fine-tuned language model (e.g., GPT-2, BART) in generating complete, coherent, and practical recipes from a diverse list of grocery items?
* To what extent can transformer-based models generalize to novel ingredient combinations or recipe types not explicitly encountered during their training on the RecipeNLG dataset?
* How do different prompt structures (e.g., including cooking context, cuisine hints, or just raw ingredient lists) influence the quality, specificity, and creativity of the generated recipes?
* (Planned Future Work) Does incorporating advanced techniques like ingredient embeddings or clustering (e.g., to understand "butter vs. margarine") improve semantic matching, ingredient substitution logic, and overall recipe quality?

## Project Structure
project-root/
├── src/
│   ├── data_loader.py            # Loads and preprocesses the RecipeNLG dataset.
│   ├── model_runner.py           # Manages loading trained models, running inference, and saving generated outputs.
│   ├── train.py                  # Contains the training logic for fine-tuning generative models.
│   └── utils/
│       └── helpers.py            # Placeholder for general utility functions.
├── configs/
│   └── model_config.yaml         # Centralized configuration file for model parameters and paths.
├── outputs/
│   ├── generated_recipes.txt     # Stores the text output of generated recipe samples.
│   └── images/                   # Placeholder for any image outputs.
├── data/
│   └── raw/                      # Directory to store the raw 'full_dataset.csv' downloaded from Kaggle.
├── notebooks/
│   └── demo_pipeline.ipynb       # Jupyter notebook for interactive demonstrations.
├── Dockerfile                    # Container definition.
├── requirements.txt              # Python dependencies.
└── README.md                     # This file.

## Setup Instructions
To set up and run this project, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/alliselwah/IE7374_Group4.git](https://github.com/alliselwah/IE7374_Group4.git)
    cd IE7374_Group4 # Navigate into the project root directory
    ```

2.  **Download the Dataset:**
    Go to the [RecipeNLG Dataset on Kaggle](https://www.kaggle.com/datasets/saldenisov/recipenlg).
    Download the `full_dataset.csv` file.
    Place `full_dataset.csv` into the `data/raw/` directory within your cloned project-root.
    (The full path should be `project-root/data/raw/full_dataset.csv`).

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Model Usage and Reproduction
This section guides you through training a model and generating recipes.

#### 1. Configure Model Parameters (`configs/model_config.yaml`)
Before running any scripts, inspect and modify `configs/model_config.yaml` to set your desired model type, training parameters, and inference settings.

Key parameters to consider:

* `model_type`: Choose between `"gpt2_pt"` (PyTorch GPT-2), `"gpt2_tf"` (TensorFlow GPT-2), or `"bart"`.
* `training_epochs`: Number of epochs for fine-tuning. For initial testing, 1 or 2 is recommended.
* `training_batch_size`: Batch size for training.
* `fine_tuned_model_path`: Crucially, after training, update this path in `model_config.yaml` to point to the saved checkpoint of your trained model (e.g., `./model_checkpoints/gpt2-pytorch/checkpoint-XXX/`). If left as `null` or an invalid path, the `model_runner.py` will load the base pre-trained model.
* `max_generation_length`: Maximum length of the generated recipe instructions.
* `num_beams`: For beam search during generation; higher values (e.g., 4 or 5) generally yield better quality.
* `num_inference_samples`: Number of recipes to generate from the validation set for demonstration.

#### Fine-tuning a Model (`train.py`)
```bash
python3 src/train.py

