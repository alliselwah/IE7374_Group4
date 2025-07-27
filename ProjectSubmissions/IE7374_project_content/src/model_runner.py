# project-root/src/model_runner.py

import torch
import tensorflow as tf # Required for TFGPT2LMHeadModel, even if not directly used by default
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, TFGPT2LMHeadModel, # GPT-2 models (PyTorch and TensorFlow)
    BartTokenizer, BartForConditionalGeneration         # BART models
)
import os
import yaml # Used for loading configuration parameters from YAML files
from data_loader import load_and_preprocess_data # Custom utility to load and prepare dataset

def load_config(config_path="configs/model_config.yaml"):
    """
    Loads configuration parameters from a YAML file.
    Args:
        config_path (str): The file path to the YAML configuration file.
    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_recipes(model, tokenizer, input_prompts, model_type: str, max_length: int = 256, num_beams: int = 4):
    """
    Generates recipe instructions based on a list of input prompts using the specified model.

    Args:
        model: The loaded Hugging Face model instance (fine-tuned or base).
        tokenizer: The tokenizer corresponding to the loaded model.
        input_prompts (list): A list of text strings, where each string is a recipe prompt
                              (e.g., "Title: ...; Ingredients: ...").
        model_type (str): The type of model being used: 'gpt2_pt' (PyTorch GPT-2),
                          'gpt2_tf' (TensorFlow GPT-2), or 'bart'.
        max_length (int): The maximum total length of the generated sequence (prompt + generated text).
        num_beams (int): The number of beams to use for beam search decoding. Higher values
                         can lead to higher quality but slower generation.

    Returns:
        list: A list of generated recipe texts, where each item corresponds to an input prompt.
    """
    generated_recipes = []
    # Determine the appropriate device (GPU 'cuda'/'mps' or CPU) for model operations.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    for prompt in input_prompts:
        if model_type.startswith("gpt2"):
            # For GPT-2 (a causal language model), the prompt is part of the sequence.
            # Encode the prompt and move it to the determined device.
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # Generate the continuation of the prompt.
            generation_output = model.generate(
                input_ids,
                max_length=max_length,         # Max length of the *entire* generated sequence.
                num_beams=num_beams,           # Use beam search for higher quality.
                no_repeat_ngram_size=2,        # Prevent repetition of 2-grams.
                early_stopping=True,           # Stop generation when an EOS token is produced.
                pad_token_id=tokenizer.eos_token_id # Crucial for handling padded batches and stopping.
            )
            # Decode the complete generated sequence (which includes the original prompt).
            decoded_output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            
            # Extract only the newly generated instructions by removing the original prompt part.
            generated_text = decoded_output[len(prompt):].strip()

        elif model_type == "bart":
            # For BART (an encoder-decoder model), the prompt is fed to the encoder.
            # Encode the prompt and prepare inputs for the model.
            inputs = tokenizer(
                prompt,
                max_length=max_length, # Max length for the encoder input.
                truncation=True,       # Truncate if prompt exceeds max_length.
                return_tensors="pt"    # Return PyTorch tensors.
            ).to(device)

            # Generate the sequence from the decoder, conditioned on encoder outputs.
            generation_output = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"], # Pass attention mask to handle padding.
                max_length=max_length, # Max length for the decoder's generated output.
                num_beams=num_beams,
                early_stopping=True,
                # BART's generation handles its own special tokens; specific `pad_token_id` is less common here.
            )
            # Decode the generated sequence from the decoder's output.
            generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        else:
            # Raise an error if an unsupported model type is specified in the configuration.
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'gpt2_pt', 'gpt2_tf', or 'bart'.")

        generated_recipes.append(generated_text)
    return generated_recipes


# --- Main Execution Block ---
# This block runs only when model_runner.py is executed directly.
if __name__ == '__main__':
    # Load configuration settings from `model_config.yaml`.
    config = load_config()

    # Extract specific parameters from the loaded configuration.
    dataset_path = config["dataset_path"]
    output_dir = config["output_dir"]
    model_type = config["model_type"]
    fine_tuned_model_path = config["fine_tuned_model_path"] # Path to the specific fine-tuned checkpoint.
    max_generation_length = config["max_generation_length"]
    num_beams = config["num_beams"]
    num_inference_samples = config["num_inference_samples"]

    # Ensure the directory for saving generated outputs exists.
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading for Inference Prompts ---
    # Construct the full path to the raw dataset file.
    full_dataset_path = os.path.join(dataset_path, "full_dataset.csv")
    
    # Verify that the dataset file exists to prevent errors.
    if not os.path.exists(full_dataset_path):
        print(f"Error: full_dataset.csv not found at {full_dataset_path}. Please provide the correct path to your dataset.")
        print("Ensure 'full_dataset.csv' is located in the 'project-root/data/raw/' directory.")
        exit() # Terminate execution if the dataset is not found.

    # Load and preprocess the data. Only the validation set is used for generating prompts here.
    _, val_df = load_and_preprocess_data(dataset_path)

    # Limit the number of samples for inference as specified in the configuration.
    if num_inference_samples > 0:
        val_df = val_df.head(num_inference_samples)

    # Extract the 'prompt' column to use as input for generation.
    input_prompts = val_df["prompt"].tolist()
    # Also extract the original 'instructions' for later comparison in the output file.
    # ground_truth_instructions = val_df["instructions"].tolist() # This comment indicates potential use.


    # --- Model and Tokenizer Loading ---
    print(f"Loading {model_type} model for inference...")
    model = None
    tokenizer = None
    # Determine the compute device to use (CUDA, MPS for Apple Silicon, or CPU).
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if model_type == "gpt2_pt":
        # Load GPT-2 tokenizer.
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Set the padding token to the end-of-sequence token for consistency in generation.
        tokenizer.pad_token = tokenizer.eos_token
        if fine_tuned_model_path:
            # Attempt to load the fine-tuned PyTorch GPT-2 model from the specified path.
            model_path_to_load = fine_tuned_model_path
            print(f"Attempting to load fine-tuned GPT-2 PyTorch model from: {model_path_to_load}")
            try:
                model = GPT2LMHeadModel.from_pretrained(model_path_to_load)
            except Exception as e:
                # Fallback to the base GPT-2 model if loading the fine-tuned one fails.
                print(f"Could not load fine-tuned model from {model_path_to_load}. Loading base GPT-2 instead. Error: {e}")
                model = GPT2LMHeadModel.from_pretrained("gpt2")
        else:
            # Load the base GPT-2 model if no fine-tuned path is provided.
            print("No fine_tuned_model_path specified. Loading base GPT-2 PyTorch model.")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.to(device) # Move the PyTorch model to the selected device.

    elif model_type == "gpt2_tf":
        # Load GPT-2 tokenizer for TensorFlow.
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        if fine_tuned_model_path:
            # Attempt to load the fine-tuned TensorFlow GPT-2 model.
            model_path_to_load = fine_tuned_model_path
            print(f"Attempting to load fine-tuned GPT-2 TensorFlow model from: {model_path_to_load}")
            try:
                # `from_pt=True` is useful if the TF model was originally saved from a PyTorch Trainer checkpoint.
                model = TFGPT2LMHeadModel.from_pretrained(model_path_to_load, from_pt=True)
            except Exception as e:
                # Fallback to the base TF GPT-2 model if loading fails.
                print(f"Could not load fine-tuned model from {model_path_to_load}. Loading base GPT-2 TF instead. Error: {e}")
                model = TFGPT2LMHeadModel.from_pretrained("gpt2")
        else:
            # Load the base TensorFlow GPT-2 model.
            print("No fine_tuned_model_path specified. Loading base GPT-2 TensorFlow model.")
            model = TFGPT2LMHeadModel.from_pretrained("gpt2")
        # TensorFlow models automatically handle device placement, so explicit .to(device) is not needed.

    elif model_type == "bart":
        # Load BART tokenizer.
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        if fine_tuned_model_path:
            # Attempt to load the fine-tuned BART model.
            model_path_to_load = fine_tuned_model_path
            print(f"Attempting to load fine-tuned BART model from: {model_path_to_load}")
            try:
                model = BartForConditionalGeneration.from_pretrained(model_path_to_load)
            except Exception as e:
                # Fallback to the base BART model if loading fails.
                print(f"Could not load fine-tuned model from {model_path_to_load}. Loading base BART instead. Error: {e}")
                model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        else:
            # Load the base BART model.
            print("No fine_tuned_model_path specified. Loading base BART model.")
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        model.to(device) # Move the PyTorch BART model to the selected device.

    else:
        # Raise an error if an unsupported model type is specified in the configuration.
        raise ValueError(f"Invalid model_type in config: {model_type}. Choose 'gpt2_pt', 'gpt2_tf', or 'bart'.")

    # Confirm that a model and tokenizer were successfully loaded before proceeding.
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        exit()

    print("Model and tokenizer loaded successfully.")
    print(f"Using device: {device}") # Indicate which device the model is running on.

    # --- Generate Recipes ---
    print(f"\nGenerating {len(input_prompts)} recipes...")
    generated_recipes = generate_recipes(
        model, tokenizer, input_prompts, model_type, max_generation_length, num_beams
    )

    # --- Save Results to File ---
    output_filepath = os.path.join(output_dir, "generated_recipes.txt")
    with open(output_filepath, "w") as f:
        # Write each generated recipe along with its prompt and original instructions for comparison.
        for i, (prompt, generated_text, original_instructions) in enumerate(zip(input_prompts, generated_recipes, val_df["instructions"].tolist())):
            f.write(f"--- Recipe {i+1} ---\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated Instructions: {generated_text}\n")
            f.write(f"Original Instructions: {original_instructions}\n") # Original is for reference.
            f.write("-" * 50 + "\n\n") # Separator for readability.

    print(f"\nGenerated recipes saved to: {output_filepath}")

    # --- Optional: Print Sample Generated Recipes to Console ---
    # This provides a quick preview without needing to open the output file.
    print("\n--- Sample Generated Recipes (Console Output) ---")
    for i, (prompt, generated_text) in enumerate(zip(input_prompts, generated_recipes)):
        if i >= 3: # Limit the console output to the first 3 samples.
            break
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}\n")