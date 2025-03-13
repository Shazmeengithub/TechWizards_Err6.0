# Import libraries
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Load the model and tokenizer
def load_llama_model():
    """
    Load the Llama3 8B model and tokenizer with 4-bit quantization.
    """
    print("Loading Llama3 8B model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",  # Base model
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

# Apply LoRA fine-tuning setup
def configure_lora(model):
    """
    Apply LoRA (Low-Rank Adaptation) to the model for parameter-efficient fine-tuning.
    """
    print("Configuring LoRA for fine-tuning...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing=True,
        random_state=42,
    )
    print("LoRA configuration complete!")
    return model

# Load the training dataset
def load_medical_dataset():
    """
    Load the medical dataset for fine-tuning.
    """
    print("Loading medical dataset...")
    dataset_train = load_dataset("sathvik123/llama3-medical-dataset", split="train")
    print(f"Dataset loaded with {len(dataset_train)} samples.")
    return dataset_train

# Fine-tune the model
def train_model(model, tokenizer, dataset_train):
    """
    Fine-tune the model using the SFTTrainer from the `trl` library.
    """
    print("Starting fine-tuning...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        dataset_text_field="prompt",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Disable packing for better control over sequence lengths
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="fine_tuned_model",  # Local directory to save training outputs
            report_to="none",  # Disable reporting to external services
        ),
    )
    trainer_stats = trainer.train()
    print("Fine-tuning completed!")
    return trainer_stats

# Save the fine-tuned model locally
def save_model(model, tokenizer, save_dir="fine_tuned_model"):
    """
    Save the fine-tuned model and tokenizer to a local directory.
    """
    print(f"Saving model and tokenizer to '{save_dir}'...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved successfully in '{save_dir}'!")

# Main function to execute the fine-tuning pipeline
def main():
    # Step 1: Load the model and tokenizer
    model, tokenizer = load_llama_model()

    # Step 2: Apply LoRA fine-tuning setup
    model = configure_lora(model)

    # Step 3: Load the training dataset
    dataset_train = load_medical_dataset()

    # Step 4: Fine-tune the model
    train_model(model, tokenizer, dataset_train)

    # Step 5: Save the fine-tuned model locally
    save_model(model, tokenizer)

# Run the main function
if __name__ == "__main__":
    main()