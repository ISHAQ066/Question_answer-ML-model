# Efficient 4-Bit Quantization and Fine-Tuning of LLaMA 2 for Question Answering

This repository contains code for fine-tuning a 4-bit quantized version of the LLaMA 2 (7B) language model for efficient question answering. The project leverages `bitsandbytes` for quantization, PEFT with LoRA for parameter-efficient fine-tuning, and integrates a custom evaluation framework for testing the model with custom questions and contexts.

## Project Overview

The goal of this project is to optimize the LLaMA 2 (7B) model using 4-bit quantization to reduce computational overhead while maintaining high performance in domain-specific question answering tasks. The project involves fine-tuning the model on the SQuAD dataset and implementing a robust evaluation system.

## Technologies Used

- **Hugging Face Transformers**: Model loading, fine-tuning, and evaluation
- **`bitsandbytes`**: 4-bit model quantization for efficient training
- **PEFT (Parameter-Efficient Fine-Tuning)**: Attaching LoRA adapters for fine-tuning
- **LoRA (Low-Rank Adaptation)**: Adding trainable adapters for efficient tuning
- **PyTorch**: Model training and tensor operations
- **CUDA**: GPU acceleration for training and inference
- **SQuAD Dataset**: Training and evaluating the QA model
- **Python**: Scripting and implementation

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/efficient-qa-llama2.git
    cd efficient-qa-llama2
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

    Ensure your `requirements.txt` includes:

    ```txt
    transformers
    datasets
    accelerate
    bitsandbytes
    torch
    peft
    ```

3. Download and prepare the SQuAD dataset:

    ```bash
    python prepare_data.py
    ```

## Usage

### Fine-Tuning the Model

To fine-tune the model, run:

```bash
python fine_tune.py
