# project-root/src/train.py

import torch
import tensorflow as tf # Keep this as TFGPT2LMHeadModel might be used for TF model variant
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, TFGPT2LMHeadModel, # GPT-2 models (PyTorch and TensorFlow)
    BartTokenizer, BartForConditionalGeneration,       # BART models
    Trainer, TrainingArguments,                        # Core components for PyTorch training loop
    TextDataset, DataCollatorForLanguageModeling      # Utilities for GPT-2 data handling
)
import os
import pandas as pd # Used for DataFrame operations, specifically for passing train_df, val_df

# --- Custom PyTorch Dataset for BART Model ---
# This class wraps tokenized data to be compatible with PyTorch's DataLoader
# and Hugging Face's Trainer for encoder-decoder models like BART.
class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets):
        """
        Initializes the dataset with tokenized input prompts and target instructions.
        Args:
            encodings: Tokenized input prompts (encoder inputs).
            targets: Tokenized target instructions (decoder labels).
        """
        self.encodings = encodings
        self.targets = targets

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample by index.
        Returns a dictionary formatted for BART: input_ids (encoder), attention_mask, and labels (decoder).
        """
        return {
            "input_ids": self.encodings.input_ids[idx],
            "attention_mask": self.encodings.attention_mask[idx],
            "labels": self.targets.input_ids[idx] # Labels are the target sequence for BART
        }

# --- Data Preparation Utility for GPT-2 (TextDataset) ---
# This function creates temporary text files, which TextDataset (a legacy utility)
# reads from directly. This is specific to how TextDataset works.
def prepare_gpt2_data_files(train_texts, val_texts, tokenizer, max_length=512):
    """
    Prepares temporary text files from lists of strings for GPT-2's TextDataset.
    Args:
        train_texts (list): List of concatenated prompt+instructions for training.
        val_texts (list): List of concatenated prompt+instructions for validation.
        tokenizer: The tokenizer to associate with the dataset.
        max_length (int): The block size for tokenizing text in TextDataset.
    Returns:
        tuple: (train_dataset, val_dataset) as Hugging Face TextDataset objects.
    """
    # Create a temporary directory to store the text files.
    temp_dir = "./temp_gpt2_data"
    os.makedirs(temp_dir, exist_ok=True)

    # Define paths for temporary train and validation files.
    train_file = os.path.join(temp_dir, "gpt2_train.txt")
    val_file = os.path.join(temp_dir, "gpt2_val.txt")

    # Write training texts to the temporary training file.
    with open(train_file, "w") as f:
        for line in train_texts:
            f.write(line + "\n")
    # Write validation texts to the temporary validation file.
    with open(val_file, "w") as f:
        for line in val_texts:
            f.write(line + "\n")

    # Initialize TextDataset for both training and validation.
    # TextDataset automatically handles reading from the file and tokenizing based on block_size.
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=max_length
    )
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=val_file,
        block_size=max_length
    )
    
    # Clean up: Remove the temporary text files immediately after dataset creation.
    os.remove(train_file)
    os.remove(val_file)
    # Clean up: Remove the temporary directory if it becomes empty.
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    return train_dataset, val_dataset

# --- Main Model Training Function ---
def train_model(model_name: str, train_df, val_df, output_dir="./model_checkpoints", epochs=3, batch_size=4):
    """
    Orchestrates the training process for a specified generative model.
    Supports GPT-2 (PyTorch/TensorFlow) and BART models.

    Args:
        model_name (str): Specifies the model to train ('gpt2_pt', 'gpt2_tf', or 'bart').
        train_df (pd.DataFrame): DataFrame containing training data.
        val_df (pd.DataFrame): DataFrame containing validation data.
        output_dir (str): Base directory where trained models/checkpoints will be saved.
        epochs (int): Number of training epochs (full passes over the training data).
        batch_size (int): Number of samples per batch during training.

    Returns:
        tuple: The trained model object and its corresponding tokenizer.
    """
    # --- GPT-2 PyTorch Training Logic ---
    if model_name == "gpt2_pt":
        # Load GPT-2 tokenizer and model from Hugging Face Hub.
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Set pad token to EOS token; crucial for GPT-2 batching and generation termination.
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Prepare text data for GPT-2 by concatenating prompt and instructions.
        # This creates the causal language modeling objective for GPT-2.
        train_texts = (train_df["prompt"] + "\n" + train_df["instructions"]).tolist()
        val_texts = (val_df["prompt"] + "\n" + val_df["instructions"]).tolist()

        # Create TextDataset objects from the prepared texts.
        train_dataset, val_dataset = prepare_gpt2_data_files(train_texts, val_texts, tokenizer)

        # Configure TrainingArguments for the Hugging Face Trainer.
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "gpt2-pytorch"), # Path to save checkpoints for this specific model.
            overwrite_output_dir=True,                           # Overwrite previous checkpoints if they exist.
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_steps=1000,                                     # Frequency of saving model checkpoints.
            save_total_limit=2,                                  # Maximum number of checkpoints to keep.
            eval_strategy="steps",                               # Evaluation performed at 'logging_steps'.
            logging_dir='./logs',                                # Directory for TensorBoard logs.
            logging_steps=500,                                   # Frequency of logging training metrics.
            report_to="tensorboard"                              # Enables logging to TensorBoard.
        )

        # Initialize the Hugging Face Trainer.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # Data collator for causal language modeling (mlm=False ensures no masked language modeling).
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        # Start the training process.
        trainer.train()
        return model, tokenizer

    # --- GPT-2 TensorFlow Training Logic ---
    elif model_name == "gpt2_tf":
        # Check for `tf_keras` which is often needed for TensorFlow models in modern environments.
        try:
            import tf_keras as keras # Aliased for compatibility
        except ImportError:
            print("tf_keras not found. Please install it with 'pip install tf-keras' to use GPT-2 TensorFlow model.")
            raise

        # Load GPT-2 tokenizer and TensorFlow GPT-2 model.
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = TFGPT2LMHeadModel.from_pretrained("gpt2")

        # Tokenize input prompts and instructions, returning TensorFlow tensors.
        inputs = tokenizer(
            (train_df["prompt"] + "\n" + train_df["instructions"]).tolist(),
            return_tensors="tf", # Specify TensorFlow tensor output
            padding=True,        # Pad sequences to the longest in the batch
            truncation=True,     # Truncate sequences longer than max_length
            max_length=512       # Maximum sequence length for tokenization
        )

        # Ensure the output directory for TensorFlow model checkpoints exists.
        os.makedirs(os.path.join(output_dir, "gpt2-tensorflow"), exist_ok=True)

        # Compile the TensorFlow model with an optimizer for training.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5))
        
        # Train the model using TensorFlow's Keras-style .fit() method.
        # For causal language modeling, input_ids are often used as both features and labels.
        model.fit(inputs["input_ids"], inputs["input_ids"], epochs=epochs, batch_size=batch_size)

        # Save the fine-tuned TensorFlow model and its tokenizer.
        model.save_pretrained(os.path.join(output_dir, "gpt2-tensorflow"))
        tokenizer.save_pretrained(os.path.join(output_dir, "gpt2-tensorflow"))

        return model, tokenizer

    # --- BART Training Logic ---
    elif model_name == "bart":
        # Load BART tokenizer and model.
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

        # Tokenize prompts for BART's encoder input.
        inputs = tokenizer(
            train_df["prompt"].tolist(),
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # Tokenize instructions for BART's decoder input (these will be the labels for training).
        targets = tokenizer(
            train_df["instructions"].tolist(),
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Create a custom RecipeDataset for BART, which handles encoder-decoder input format.
        dataset = RecipeDataset(inputs, targets)

        # Configure TrainingArguments for BART training.
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "bart"), # Specific output directory for BART checkpoints.
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=100,
            save_steps=1000,
            eval_strategy="steps", # Evaluation performed based on steps.
            logging_dir='./logs',
        )

        # Initialize the Trainer for BART.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset # Use the custom dataset for BART
        )

        # Start the training process for BART.
        trainer.train()
        return model, tokenizer
    else:
        # Raise an error if an unsupported model name is provided.
        raise ValueError("Invalid model_name. Choose 'gpt2_pt', 'gpt2_tf', or 'bart'.")

# --- Main Execution Block ---
# This block runs only when train.py is executed directly (not when imported as a module).
if __name__ == '__main__':
    # Import the data loading utility.
    from data_loader import load_and_preprocess_data

    # Define the path to your raw dataset, relative to the 'project-root' directory.
    DATASET_PATH = "data/raw/"

    # Check if the 'full_dataset.csv' file exists at the specified path.
    # This prevents errors if the dataset is missing.
    if not os.path.exists(os.path.join(DATASET_PATH, "full_dataset.csv")):
        print(f"Error: full_dataset.csv not found at {DATASET_PATH}. Please provide the correct path to your dataset.")
        print("Ensure 'full_dataset.csv' is located in the 'project-root/data/raw/' directory.")
        exit() # Terminate the script if the data file is not found.

    # Load and preprocess the actual dataset.
    # Note: For full training, remove `.head(X)` from train_df and val_df below.
    train_df, val_df = load_and_preprocess_data(DATASET_PATH)

    # --- Example Training Calls ---
    # Uncomment the model you wish to train.
    # Adjust 'epochs' and 'batch_size' in model_config.yaml.
    # Using `.head(X)` for train_df and val_df is for *quick local testing* with a small subset.
    # For actual, effective fine-tuning, you should use the full `train_df` and `val_df`.

    # Example: Training a GPT-2 PyTorch model
    print("Starting GPT-2 PyTorch training...")
    gpt2_pt_model, gpt2_pt_tokenizer = train_model("gpt2_pt", train_df.head(50), val_df.head(10), epochs=1, batch_size=2)
    print("GPT-2 PyTorch training complete.")

    # Example: Training a BART model
    # print("\nStarting BART training...")
    # bart_model, bart_tokenizer = train_model("bart", train_df.head(50), val_df.head(10), epochs=1, batch_size=2)
    # print("BART training complete.")

    # Example: Training a GPT-2 TensorFlow model
    # print("\nStarting GPT-2 TensorFlow training...")
    # gpt2_tf_model, gpt2_tf_tokenizer = train_model("gpt2_tf", train_df.head(50), val_df.head(10), epochs=1, batch_size=2)
    # print("GPT-2 TensorFlow training complete.")