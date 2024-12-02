# Fine-tuning Pegasus for Dialogue Summarization

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

## Challanges : 

The dialogues are huge in the dataset and frequently face CUDA out of memory errors , that's we used Memory fragmentation 

What is memory fragmentation?

Imagine your GPU memory as a parking lot. When you first start, it's empty and you can park large vehicles (tensors) easily. However, as you allocate and deallocate memory (cars entering and leaving), the free space becomes fragmented into smaller, non-contiguous chunks. This makes it difficult to find a single, large space for a new, big tensor, even if there's enough total free space available. This leads to the "CUDA out of memory" error.


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

This environment variable is a way to fine-tune how PyTorch manages memory allocation on your NVIDIA GPU. It's specifically designed to address a common issue called memory fragmentation.

How does expandable_segments:True help?

This setting aims to mitigate fragmentation by enabling a feature called "expandable segments" within the PyTorch CUDA memory allocator. With expandable segments enabled, the allocator can try to merge these smaller free chunks into larger blocks when a large allocation request comes in. This reduces fragmentation and allows PyTorch to use the available GPU memory more efficiently.
