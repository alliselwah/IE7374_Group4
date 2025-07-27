# project-root/src/data_loader.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(dataset_path: str, test_size: float = 0.1, random_state: int = 42):
    """
    Loads the recipe dataset, renames columns, constructs input prompts,
    and splits the data into training and validation sets.

    Args:
        dataset_path (str): The file path to the directory containing 'full_dataset.csv'.
                            Expected to be relative to the project root (e.g., 'data/raw/').
        test_size (float): The proportion of the dataset to allocate for the validation split.
        random_state (int): A seed for the random number generator, ensuring
                            reproducible train-test splits.

    Returns:
        tuple: A tuple containing two pandas DataFrames: (train_df, val_df).
               Each DataFrame includes 'prompt' and 'instructions' columns.
    """
    # Construct the full path to the CSV file and load the dataset.
    df = pd.read_csv(os.path.join(dataset_path, "full_dataset.csv"))

    # Select the necessary columns ('title', 'NER' for ingredients, 'directions' for instructions)
    # and remove any rows where these specific columns have missing values.
    df = df[['title', 'NER', 'directions']].dropna()

    # Rename columns for clarity and consistency across the model pipeline.
    df = df.rename(columns={
        'NER': 'ingredients',
        'directions': 'instructions'
    })

    # Create the 'prompt' column by combining the recipe title and ingredients.
    # This forms the input string that the generative model will use to start generating instructions.
    df['prompt'] = "Title: " + df['title'] + "; Ingredients: " + df['ingredients']

    # Split the DataFrame into training and validation sets.
    # 'test_size' determines the size of the validation set (e.g., 0.1 means 10%).
    # 'random_state' ensures the split is the same every time the code runs.
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Return the training and validation DataFrames.
    return train_df, val_df


# This block executes only when data_loader.py is run directly (not when imported).
# It's useful for testing the data loading and preprocessing functionality in isolation.
if __name__ == '__main__':
    # Define the path to the raw dataset. This path is relative to the 'project-root'
    # when 'python3 src/data_loader.py' is executed from 'project-root'.
    DATASET_PATH = "data/raw/"

    # Load and preprocess the data using the function defined above.
    train_df, val_df = load_and_preprocess_data(DATASET_PATH)

    # Print the number of samples loaded into each set.
    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")

    # Display a sample prompt and its corresponding instructions from the training set.
    print("\nSample Training Prompt:")
    print(train_df['prompt'].iloc[0]) # .iloc[0] safely accesses the first row
    print("\nSample Training Instructions:")
    print(train_df['instructions'].iloc[0])