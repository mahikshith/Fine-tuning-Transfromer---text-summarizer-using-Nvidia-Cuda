# Fine-tuning-Transfromer - text-summarizer-using-Nvidia-Cuda

To fine-tune a pre-trained Transformer-based text summarization model on custom data using an NVIDIA CUDA-enabled GPU.

### Fine-tuning Pegasus for Dialogue Summarization

This project demonstrates how to fine-tune a pre-trained Pegasus model for dialogue summarization using the SAMSum dataset. It leverages the Hugging Face Transformers library and an NVIDIA CUDA-enabled GPU for accelerated training.

## Objective

To create a model capable of generating concise and informative summaries of dialogues. This is achieved by fine-tuning a pre-trained Pegasus model on the SAMSum dataset, a collection of dialogues with human-written summaries.

## Dataset

The project utilizes the [SAMSum dataset](https://huggingface.co/datasets/Samsung/samsum) from Hugging Face Datasets. This dataset contains dialogues and their corresponding summaries, making it suitable for training a dialogue summarization model.

## Model

The [Pegasus model](https://huggingface.co/google/pegasus-cnn_dailymail) from Google is employed as the base model for fine-tuning. Pegasus is a Transformer-based model specifically designed for abstractive text summarization, making it well-suited for this task.

## Requirements

* Python 3.7 or higher
* PyTorch
* Transformers library
* Datasets library
* Rouge Score library
* Accelerate library
* An NVIDIA CUDA-enabled GPU (recommended)

## Installation

Install the required libraries:

!pip install transformers[sentencepiece] datasets rouge_score py7zr accelerate -U -q


## Usage

1. Clone this repository to your local machine or Google Colab environment.
2. Download the SAMSum dataset using the `datasets` library

3. Fine-tune the Pegasus model using the provided training script (`train.py`). This script defines the training parameters, data loading, and fine-tuning process.
4. Evaluate the model on the test set using the evaluation script (`evaluate.py`). This script calculates the ROUGE scores to measure the model's performance.
5. Use the fine-tuned model for inference on new dialogues.
