# IE 7374 Group Project

Group 4 Members: 
- Akshit Sharma
- Allison Selwah
- Anna Grondolsky
- Karthik Veluchamy Sarguru

# Project Proposal

## Project Title & Description

**A Generative AI Tool for Recipe Creation Based on Available Household Ingredients**

A generative AI-powered tool designed to help users create meal ideas from ingredients they already have at home. By inputting a list of grocery items, users receive one or more recipe suggestions that incorporate those items effectively. The system leverages natural language processing and generative models to craft complete and coherent recipes including ingredients lists, instructions, and preparation times. This tool aims to reduce food waste, save time, and inspire home cooking using artificial intelligence.

---

## Final Topic Area

We will be working in the field of Natural Language Processing (NLP) for our final project. Our chosen topic involves generating simple, easy-to-make recipes based on natural language ingredient inputs. The rationale behind this choice stems from our interest in how machines can understand and generate human language, particularly in a creative and practical domain like cooking.

This project emphasizes sequence generation and semantic understanding, both core NLP tasks we’ve explored in this course. By converting a list of ingredients into coherent and useful recipe steps, we aim to apply NLP techniques such as tokenization, embedding, and potentially transformer-based models. The challenge and excitement lie in teaching a model to grasp not just words, but the contextual and procedural knowledge they imply — something we believe showcases the real-world potential of NLP.

---

## Dataset Description

For our project, we are using the **Recipe1M+** dataset, developed by MIT CSAIL and available on Kaggle (RecipeNLG Dataset). This dataset contains approximately one million recipes, each consisting of a title, a list of ingredients, and detailed cooking instructions. While the full dataset includes images, we will focus exclusively on the textual components, as our work centers on NLP.

The data is stored in JSON format, which makes it easy to parse and manipulate for training language models. Each recipe is cleanly structured, which is especially helpful for tasks like sequence generation. We are particularly interested in the ingredients-to-instructions relationship, as our goal is to generate simple recipes based on user-provided ingredient lists — mimicking the real-world scenario of deciding what to cook based on what’s in the fridge.

One key advantage of this dataset is its scale and diversity, which will allow us to train or fine-tune a generative model to produce coherent and relevant cooking steps. We also plan to simulate user inputs by extracting subsets of ingredients from existing recipes, allowing the model to learn how to generate plausible instructions from partial ingredient lists.

Overall, Recipe1M+ offers the structure, volume, and content richness we need to support our project’s goals in natural language generation and semantic understanding.

---

## Model Selection

For our project, we plan to use a GPT-based generative language model, specifically **GPT-2**, which is a transformer-based decoder architecture developed by OpenAI.

GPT-2 is well-suited for our task because it excels at natural language generation and can produce coherent, contextually relevant text based on a given input prompt. Our research focuses on generating step-by-step recipe instructions from a list of ingredients, a classic sequence-to-sequence problem where understanding the semantics of the input and generating structured output is crucial. GPT-2’s architecture, designed for left-to-right language modeling, aligns perfectly with the task of generating fluent, ordered recipe instructions.

We will use a pre-trained version of GPT-2 (likely via the Hugging Face Transformers library) and fine-tune it on the Recipe1M+ dataset. This approach is recommended because:

- It leverages existing linguistic knowledge already learned by GPT-2, reducing training time and computational cost.  
- Fine-tuning allows us to adapt the model to the cooking domain, learning patterns specific to ingredients and cooking instructions.  
- Using a pretrained model also enables us to focus on designing input representations (e.g., formatting the ingredient list as a prompt) and evaluating output quality rather than training a large model from scratch.

Overall, GPT-2 offers the right balance of performance, accessibility, and flexibility for generating natural-sounding recipes from textual inputs.

---

## Research Questions

- **How effective is a fine-tuned language model (e.g., GPT-2) in generating complete, coherent recipes from a random list of grocery items?**  
  This evaluates the model’s ability to structure logical and readable cooking instructions from unordered and variable ingredient inputs.

- **Can transformer-based models generalize to novel ingredient combinations not seen during training?**  
  We want to test the model's creativity and semantic understanding when faced with unfamiliar inputs — mimicking real-world use cases where pantry contents are unpredictable.

- **How do different prompt structures (e.g., including cooking context, cuisine hints, or just raw ingredient lists) affect the quality and specificity of generated recipes?**  
  By comparing prompt variations, we aim to identify which formats guide the model toward more useful and coherent outputs.

- **Does incorporating ingredient embeddings or clustering improve semantic matching and substitution (e.g., butter vs. margarine)?**  
  This investigates whether embedding-based techniques can help the model recognize functionally similar ingredients, enhancing its flexibility and output quality.

---

## Plan of Action

### Data Preprocessing

- Clean and organize the Recipe1M+ dataset:  
  - Extract and normalize ingredient lists and corresponding instructions  
  - Remove incomplete or poorly formatted entries  
  - Standardize units and ingredient naming to reduce variability  
  - Simulate user inputs by creating subsets of ingredients (e.g., 3-7 items per list) to reflect real-world grocery scenarios  

### Model Implementation

- Use a pretrained GPT-2 model from the Hugging Face Transformers library.  
- Fine-tune the model on the cleaned dataset, using structured prompts to encourage coherent recipe generation.  
- Experiment with prompt engineering, including variations like adding context or cuisine tags to guide outputs.

### Experimental Design & Evaluation

- Conduct multiple experiments to test different configurations:  
  - Baseline: Raw ingredient list as prompt  
  - Context-enhanced: Prompts with cuisine type or desired dish name  
  - Embedding-aided: Using pretrained ingredient embeddings to improve semantic matching  

- Evaluation metrics:  
  - BLEU or ROUGE scores to measure similarity to ground truth instructions  
  - Perplexity to assess model fluency  
  - Human evaluation (if time allows) to assess readability and usefulness  

### Analysis Techniques

- Compare performance across prompt types and ingredient subsets.  
- Use clustering to analyze ingredient embedding quality.  
- Visualize attention weights or embedding distances to understand model behavior.

---

## Team Contribution

Our team consists of five defined roles, with responsibilities distributed based on expertise and availability. Two members will lead technical development, while others contribute to evaluation, testing, and documentation. All team members share responsibility for communication, planning, and final deliverables.

- **Akshit:**  
  - Fine-tune the GPT-2 model using Hugging Face Transformers.  
  - Implement and experiment with prompt engineering techniques.  
  - Oversee model training and optimization processes.

- **Allison:**  
  - Clean and preprocess the Recipe1M+ dataset.  
  - Normalize ingredients and create structured prompts.  
  - Generate training and test samples based on real-world input simulation.

- **Anna:**  
  - Design and apply evaluation metrics (BLEU, ROUGE, perplexity).  
  - Conduct error analysis and identify model weaknesses.  
  - Help interpret both quantitative and qualitative results.

- **Karthik:**  
  - Document all preprocessing, model, and evaluation steps.  
  - Assist with user simulation testing and prompt response validation.  
  - Lead preparation of the final report and visual materials for presentation.

- **All Members:**  
  - Participate in weekly check-ins and milestone reviews.  
  - Review and test intermediate model outputs.  
  - Contribute to discussions of findings, final recommendations, and course-aligned reflections.  
  - Ensure code reproducibility, shared repositories, and version control practices are followed.

# Data Pipeline

## Research and Selection of Methods

### Core NLP Task

The core task involves generating step-by-step recipe instructions from a structured input list of ingredients. We frame this as a text-to-text generation problem, where the model learns to map structured inputs like the following:

  'Input: "Title: Spicy Tofu Stir Fry; Ingredients: tofu, chili sauce, garlic, soy sauce, vegetables"'
  'Target: "Cut the tofu into cubes and fry until golden. Add chopped vegetables and sauté. Mix in chili sauce, soy sauce, and garlic..."'


### Secondary NLP Tasks

- **Prompt Engineering:** Used during early experimentation with large language models (like GPT-2) to generate coherent recipe steps based on structured prompt formats.  
  Example prompt:  
  `"Title: Banana Bread; Ingredients: banana, flour, sugar, eggs. Generate instructions."`

- **Embedding and Similarity Matching:** Used sentence-level embeddings (via `sentence-transformers`) to semantically represent individual ingredients or ingredient lists. We leveraged **FAISS** to find and suggest:  
  - Similar ingredients (e.g., beef → tofu)  
  - Ingredient clustering or replacement suggestions for dietary needs

- **Text Generation Evaluation:**  
  - **BLEU score:** N-gram overlap with ground truth instructions  
  - **ROUGE:** Recall-based evaluation of instruction similarity  
  - Manual evaluation was also performed to assess fluency and coherence.

### Literature Review

#### Model Architectures

- **T5 (Text-to-Text Transfer Transformer):** Fine-tuned to map structured inputs (e.g., title and ingredients) to coherent instructions using Hugging Face’s transformers and Trainer API.  
- **BART:** Used alongside T5; performs well on text summarization and generation tasks, also fine-tuned with Hugging Face.  
- **GPT-style Models (GPT-2):** Used during initial exploration for prompt-based generation, good for zero-shot and few-shot learning but expensive to deploy at scale.

#### Dataset Referenced

Our primary dataset was the **RecipeNLG** dataset from Kaggle (about 2 million recipes), utilizing CSV fields to construct model inputs and ground-truth outputs.

Originally, we considered the **Recipe1M+** dataset (specifically the `layer2+.json` file) due to its rich structure and scale, but RecipeNLG was chosen for cleaner formatting and easier preprocessing.

### Benchmarking

| Model         | Strengths                              | Tradeoffs                   |
| ------------- | ------------------------------------ | -------------------------- |
| T5 / BART     | Natively supports structured-to-text | May require input format tuning |
| GPT (API)     | Good with prompt-based input          | API cost and token limits   |
| Sentence-BERT | Easy to compute similarity for substitutions | External to generation pipeline |

### Preliminary Experiments

Initial experiments involved creating prompt templates using the title and ingredients from RecipeNLG entries to generate recipe instructions. Early evaluation using BLEU scores compared generated outputs to ground truth. These tests confirmed that RecipeNLG offers a cleaner and more consistent dataset compared to Recipe1M+, facilitating faster and more reliable model development and iteration.

---

## Model Implementation

### Framework Selection

The project uses the Hugging Face Transformers library to fine-tune sequence-to-sequence models such as T5 and BART on the RecipeNLG dataset. PyTorch serves as the backend framework for training and inference, providing efficient tensor computation and GPU acceleration. For ingredient similarity tasks, SentenceTransformers combined with FAISS enables embedding generation and fast nearest-neighbor searches. Text evaluation metrics like BLEU and ROUGE are computed using NLTK and related libraries, while pandas is used for flexible loading, exploration, and preprocessing of the RecipeNLG CSV files.

### Dataset Preparation

The RecipeNLG CSV dataset is loaded using pandas, selecting key fields such as recipe title, ingredients, and instructions to construct structured inputs and targets. Input sequences are created by concatenating recipe titles with comma-separated ingredient lists, forming clear prompts for the model. Both input and target texts are cleaned and tokenized using the Hugging Face tokenizer, applying truncation and padding to maintain consistent sequence lengths. The data is then split into training and validation subsets to facilitate unbiased evaluation.

### Model Development

Pre-trained transformer models, specifically T5-small or BART-based, are fine-tuned on the prepared RecipeNLG data using Hugging Face’s Trainer API, which streamlines training and evaluation. Maximum token lengths are typically set between 256 and 512 tokens for inputs and outputs to ensure sufficient context coverage while balancing computational requirements. The implementation emphasizes modularity and reusability in the codebase to allow easy experimentation with different architectures and hyperparameters.

### Training and Fine-Tuning

Initial training experiments were conducted on subsets ranging from 5,000 to 10,000 recipes, with plans to scale to larger samples as resources allow. Batch sizes between 4 and 8 were used depending on hardware, and training runs spanned 3 to 5 epochs to balance performance and training time. Training progress was monitored with periodic logging and checkpointing to enable reproducibility and recovery.

### Evaluation & Metrics

Model outputs were quantitatively assessed using BLEU scores, measuring n-gram overlap between generated recipe instructions and ground truth references. ROUGE metrics were also used to evaluate recall-based coverage of relevant instruction sequences. Additionally, manual qualitative reviews were performed to assess coherence, clarity, and culinary accuracy of generated instructions, ensuring practical usefulness beyond numerical scores.

---

## GitHub Repository Setup & Code Management

Repository: [https://github.com/alliselwah/IE7374_Group4](https://github.com/alliselwah/IE7374_Group4)
