# LLM Text Preprocessing - TDSE

Hands-on exploration of text preprocessing and tokenization fundamentals for Large Language Models, based on Chapter 2 of *"Build a Large Language Model (From Scratch)"* by Sebastian Raschka.

## Overview

This project demonstrates the essential preprocessing pipeline required to transform raw text into training data for LLMs. It covers tokenization strategies, embedding generation, and data sampling techniques that form the foundation of modern language models and agentic AI systems.

## Project Structure

```
.
├── ch2.ipynb           # Chapter 2: Working with Text Data
├── embeddings.ipynb    # Custom implementation and experiments
├── the-verdict.txt     # Sample text data (Edith Wharton short story)
└── README.md          # This file
```

## Notebooks

### `ch2.ipynb` - Working with Text Data
Original chapter notebook covering:
- Understanding word embeddings
- Text data preparation and sampling
- Visual illustrations of embedding spaces
- Code examples from the book

### `embeddings.ipynb` - Text Preprocessing Foundations
Custom experiments and implementations exploring:

1. **Loading and Preparing Text Data**
   - Understanding data quality's impact on model performance
   - Scale requirements for LLM training
   - Implications for agentic systems

2. **Tokenization**
   - Word-level vs. character-level strategies
   - Byte Pair Encoding (BPE) implementation
   - Vocabulary size tradeoffs
   - Handling rare words and domain-specific terms

3. **Training with Sliding Windows**
   - Creating training samples from raw text
   - Context window management
   - Experimenting with `max_length` and `stride` parameters
   - Understanding overlap's role in training efficiency

4. **Token Embeddings**
   - Converting tokens to dense vectors
   - How embeddings capture semantic meaning
   - Positional embeddings
   - Cosine similarity and geometric relationships

5. **Hyperparameter Experiments**
   - Impact of stride on dataset size
   - Context length tradeoffs
   - Data augmentation through overlap
   - Production considerations

## Key Concepts

### Tokenization
Tokenization bridges human language and machine learning:
- **BPE** efficiently handles rare words by breaking them into subword units
- Balances sequence length with vocabulary size
- No "unknown token" problem for new words
- Critical for handling code, URLs, and multilingual text

### Sliding Windows
LLMs learn next-token prediction using overlapping context windows:
- Each sample teaches the model to predict token N+1 given tokens 1 to N
- Overlapping windows create more training examples from limited data
- Models learn position-dependent patterns and long-range dependencies
- Foundation for multi-turn conversation understanding

### Embeddings
Dense vector representations that capture semantic meaning:
- Similar words cluster in embedding space
- Learned during training through backpropagation
- Geometric relationships encode linguistic patterns
- Shared across all positions (parameter efficiency)

## Requirements

```bash
pip install torch tiktoken numpy matplotlib
```

## Usage

1. **Run Chapter 2 notebook:**
   ```bash
   jupyter notebook ch2.ipynb
   ```
   Execute all cells to explore the original chapter content.

2. **Explore custom experiments:**
   ```bash
   jupyter notebook embeddings.ipynb
   ```
   Run experiments with different hyperparameters and analyze their impact.

## Dataset

The project uses `the-verdict.txt`, a short story by Edith Wharton, as sample text data for demonstrating:
- Text loading and preprocessing
- Tokenization strategies
- Embedding generation
- Training sample creation

## Learning Objectives

After working through these notebooks, you will understand:
- ✅ How tokenization strategies affect model performance
- ✅ The implementation of Byte Pair Encoding (BPE)
- ✅ Creating training datasets with sliding windows
- ✅ How embeddings encode semantic meaning
- ✅ The impact of hyperparameters (`max_length`, `stride`) on training
- ✅ Why these techniques matter for building agentic AI systems

## Experiments

The `embeddings.ipynb` notebook includes hands-on experiments:
- **Tokenization comparison**: Word-level vs. character-level vs. BPE
- **Sliding window impact**: How stride affects dataset size
- **Embedding visualization**: Exploring semantic relationships
- **Hyperparameter tuning**: Finding optimal `max_length` and `stride` values

## References

- **Book**: [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka
- **Author**: [Sebastian Raschka](https://sebastianraschka.com/)

## Notes

- All cells in `ch2.ipynb` have been executed
- `embeddings.ipynb` contains custom implementations and experiments
- No additional files were generated (execution only)

## License

Educational project based on materials from *Build a Large Language Model (From Scratch)* by Sebastian Raschka.