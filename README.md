# Learning Word Embeddings with Word2Vec and Neural Networks

## Overview
This project explores **Word Embeddings**, a key innovation in **Natural Language Processing (NLP)** that allows words with similar meanings to have similar vector representations. The project consists of three key parts:

- **Word2Vec (Skip-gram) Implementation**
- **Neural Network-Based Embeddings**
- **Exploring Pretrained Embeddings (GloVe)**

## Features
- Implemented Word2Vec (Skip-gram model) from scratch
- Trained word embeddings on "The Fellowship of the Ring" dataset
- Optimized using Negative Sampling
- Trained word embeddings using Neural Networks
- Explored pretrained GloVe embeddings for real-world insights
- Visualized word relationships with Dimensionality Reduction (SVD)
- Analyzed Bias in Word Embeddings

## Methodology
### 1. Word2Vec (Skip-gram) Implementation
- **Dataset:** "The Fellowship of the Ring" (Subset)
- **Preprocessing:** Tokenization, punctuation removal, lowercasing
- **Training:** Implemented Skip-gram model with Negative Sampling
- **Loss Function:** Cross-entropy loss with gradient descent
- **Evaluation:** Cosine similarity between word embeddings

### 2. Neural Network-Based Embeddings
- **Architecture:** Single hidden layer with tanh activation
- **Training:** Softmax output layer for word classification
- **Optimization:** Gradient Descent & Backpropagation
- **Performance:** Loss reduction over epochs, showing effective learning

### 3. Exploring Pretrained Embeddings (GloVe)
- Used GloVe embeddings (from gensim)
- Applied Dimensionality Reduction (SVD) for visualization
- Analyzed word analogies and biases in word embeddings
- Explored real-world implications of biased AI models

## Results & Insights
- Word2Vec effectively captures contextual word relationships.
- Neural Network embeddings enhance classification and semantic understanding.
- Pretrained embeddings (GloVe) contain linguistic and societal biases.
- Dimensionality Reduction techniques help visualize word relationships.

## Future Improvements
✅ Experiment with CBOW (Continuous Bag of Words) for comparison  
✅ Fine-tune embeddings on domain-specific datasets  
✅ Implement bias mitigation techniques for fairer word representations  
✅ Train embeddings on larger and more diverse datasets  

## Installation
```bash
git clone https://github.com/yourusername/word-embeddings.git
cd word-embeddings
pip install -r requirements.txt
```

## Usage
### Run Word2Vec Training
```bash
python src/word2vec.py
```

### Train Neural Network-Based Embeddings
```bash
python src/neural_embeddings.py
```

### Analyze Pretrained Embeddings
```bash
jupyter notebook notebooks/pretrained_glove.ipynb
```

## Dependencies
- Python 3.8+
- NumPy
- Matplotlib
- NLTK
- Scikit-learn
- TensorFlow/Keras
- PyTorch
- Gensim
- Jupyter Notebook

## Contributors
- **[Your Name]** - [GitHub Profile](https://github.com/yourusername)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- "Speech and Language Processing" by Jurafsky & Martin
- "Efficient Estimation of Word Representations in Vector Space" - Mikolov et al.
- Jay Alammar's Blog on Word2Vec & Word Embeddings

