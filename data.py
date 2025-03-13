import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

# Load and combine multiple datasets
def load_and_combine_datasets():
    """
    Load and combine multiple medical QA datasets from Hugging Face.
    """
    print("Loading and combining datasets...")
    
    # Load the first dataset
    dataset1 = load_dataset("lavita/medical-qa-datasets", "chatdoctor_healthcaremagic", split="train")
    
    # Load the second dataset
    dataset2 = load_dataset("sathvik123/llama3-medical-dataset", split="train")
    
    # Load the third dataset (if available)
    # dataset3 = load_dataset("another_dataset_name", split="train")
    
    # Combine datasets
    combined_dataset = concatenate_datasets([dataset1, dataset2])  # Add dataset3 if available
    print(f"Combined dataset loaded with {len(combined_dataset)} samples.")
    return combined_dataset

# Convert dataset to pandas DataFrame
def convert_to_dataframe(dataset):
    """
    Convert the dataset to a pandas DataFrame for easier manipulation.
    """
    print("Converting dataset to pandas DataFrame...")
    df = dataset.to_pandas()
    return df

# Create Llama3-style prompts
def create_llama_prompt(row):
    """
    Generate a Llama3-style prompt from a row of the dataset.
    """
    return f"""<|start_header_id|>system<|end_header_id|>{row["instruction"]}<|eot_id|><|start_header_id|>user<|end_header_id|> This is the question: {row["input"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{row["output"]}<|eot_id|>"""

# Add prompts to the DataFrame
def add_prompts_to_dataframe(df):
    """
    Add Llama3-style prompts to the DataFrame.
    """
    print("Adding prompts to DataFrame...")
    df['prompt'] = df.apply(create_llama_prompt, axis=1)
    return df

# Split the dataset into train and test sets
def split_dataset(dataset):
    """
    Split the dataset into train and test sets.
    """
    print("Splitting dataset into train and test sets...")
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    return train_test_split

# Save the dataset locally
def save_dataset_locally(dataset_dict, save_dir="medical_qa_dataset"):
    """
    Save the dataset locally as a DatasetDict.
    """
    print(f"Saving dataset to '{save_dir}'...")
    dataset_dict.save_to_disk(save_dir)
    print(f"Dataset saved successfully in '{save_dir}'!")

# Main function to execute the data processing pipeline
def main():
    # Step 1: Load and combine datasets
    dataset = load_and_combine_datasets()

    # Step 2: Convert the dataset to a pandas DataFrame
    df = convert_to_dataframe(dataset)

    # Step 3: Add Llama3-style prompts to the DataFrame
    df = add_prompts_to_dataframe(df)

    # Step 4: Convert the DataFrame back to a Dataset
    dataset = Dataset.from_pandas(df)

    # Step 5: Split the dataset into train and test sets
    train_test_split = split_dataset(dataset)

    # Step 6: Create a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })

    # Step 7: Save the dataset locally
    save_dataset_locally(dataset_dict)

# Run the main function
if __name__ == "__main__":
    main()