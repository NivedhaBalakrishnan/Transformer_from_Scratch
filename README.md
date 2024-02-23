# Transformer from Scratch

This repository contains an implementation of a transformer model from scratch to predict the next character in a sequence of text. The transformer model is tested and evaluated using the Sherlock Holmes text file (10MB).

## Model Details

- **Type:** Decoder-only transformer model
- **Number of Parameters:** 1.207137 Million

## Implemented Steps

The transformer model is implemented step-by-step, including:

- Token embedding
- Position embeddings
- Multihead attention layer
- Feedforward neural network with layer normalization
- Evaluated using cross-entropy loss
- Predicting the next character with softmax as the final layer

## Usage

All hyperparameters and input/output file paths are stored in the `config.ini` file. To run and evaluate the transformer model:

1. Clone this repository:
```python
git clone https://github.com/NivedhaBalakrishnan/Transformer_from_Scratch.git
```

2. Install requirements:
```python
pip install -r requirements.txt
```

3. Run the model:
```python
python3 model.py 
```

The model will train on the Sherlock Holmes text file specified in the `config.ini` file as sherlock.txt.

Please review the generated output in the 'output.txt' file. While the generated text may not be perfect, you can potentially improve its performance by increasing the embedding dimension, number of heads, and layers in the model.
