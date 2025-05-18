# ICS2203 Statistical Language Processing - Language Model Project

### Author: Miguel Baldacchino  
📅 Date: May 2025

## 📌 Overview

This project implements a statistical language model from scratch using a subset of the **Maltese Corpus (Korpus Malti v3.0)**. The system supports **Unigram**, **Bigram**, and **Trigram** models with three variants:

- **Vanilla**: Basic Maximum Likelihood Estimation (MLE)
- **Laplace**: Adds +1 smoothing to handle zero counts
- **UNK**: Replaces rare words with `<UNK>` to improve generalization

The models support:
- Sentence generation
- Sentence probability calculation
- Perplexity evaluation
- Linear interpolation

## 🗃️ Dataset

- **Source**: Korpus Malti v3.0 ([Gatt & Čéplö, 2013])
- **Format**: `.vrt` files (verticalized, annotated text)
- **Subset**: 300 files (~2.3M sentences, ~44M tokens)

## 🛠️ Features

- Custom VRT corpus extraction and parsing
- Full preprocessing pipeline (punctuation removal, lowercasing, tokenization)
- Manual and library-based N-gram generation (with performance benchmarks)
- Weighted random sampling for sentence generation
- Unknown word handling (`<UNK>`)
- Linear interpolation with tunable weights
- Modular and scalable Python codebase
- Detailed resource monitoring (time, memory)


