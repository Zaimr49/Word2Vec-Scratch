<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning Word Embeddings</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        h1, h2 { color: #333; }
        code { background: #f4f4f4; padding: 2px 5px; }
        pre { background: #f4f4f4; padding: 10px; overflow-x: auto; }
        ul { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Learning Word Embeddings with Word2Vec and Neural Networks</h1>
    
    <h2>Overview</h2>
    <p>This project explores <strong>Word Embeddings</strong>, a key innovation in <strong>Natural Language Processing (NLP)</strong> that allows words with similar meanings to have similar vector representations. The project consists of three key parts:</p>
    <ul>
        <li><strong>Word2Vec (Skip-gram) Implementation</strong></li>
        <li><strong>Neural Network-Based Embeddings</strong></li>
        <li><strong>Exploring Pretrained Embeddings (GloVe)</strong></li>
    </ul>

    <h2>Features</h2>
    <ul>
        <li>Implemented Word2Vec (Skip-gram model) from scratch</li>
        <li>Trained word embeddings on "The Fellowship of the Ring" dataset</li>
        <li>Optimized using Negative Sampling</li>
        <li>Trained word embeddings using Neural Networks</li>
        <li>Explored pretrained GloVe embeddings for real-world insights</li>
        <li>Visualized word relationships with Dimensionality Reduction (SVD)</li>
        <li>Analyzed Bias in Word Embeddings</li>
    </ul>

    <h2>Methodology</h2>
    <h3>1. Word2Vec (Skip-gram) Implementation</h3>
    <ul>
        <li><strong>Dataset:</strong> "The Fellowship of the Ring" (Subset)</li>
        <li><strong>Preprocessing:</strong> Tokenization, punctuation removal, lowercasing</li>
        <li><strong>Training:</strong> Implemented Skip-gram model with Negative Sampling</li>
        <li><strong>Loss Function:</strong> Cross-entropy loss with gradient descent</li>
        <li><strong>Evaluation:</strong> Cosine similarity between word embeddings</li>
    </ul>

    <h3>2. Neural Network-Based Embeddings</h3>
    <ul>
        <li><strong>Architecture:</strong> Single hidden layer with tanh activation</li>
        <li><strong>Training:</strong> Softmax output layer for word classification</li>
        <li><strong>Optimization:</strong> Gradient Descent & Backpropagation</li>
        <li><strong>Performance:</strong> Loss reduction over epochs, showing effective learning</li>
    </ul>

    <h3>3. Exploring Pretrained Embeddings (GloVe)</h3>
    <ul>
        <li>Used GloVe embeddings (from gensim)</li>
        <li>Applied Dimensionality Reduction (SVD) for visualization</li>
        <li>Analyzed word analogies and biases in word embeddings</li>
        <li>Explored real-world implications of biased AI models</li>
    </ul>

    <h2>Results & Insights</h2>
    <ul>
        <li>Word2Vec effectively captures contextual word relationships.</li>
        <li>Neural Network embeddings enhance classification and semantic understanding.</li>
        <li>Pretrained embeddings (GloVe) contain linguistic and societal biases.</li>
        <li>Dimensionality Reduction techniques help visualize word relationships.</li>
    </ul>

    <h2>Future Improvements</h2>
    <ul>
        <li>✅ Experiment with CBOW (Continuous Bag of Words) for comparison</li>
        <li>✅ Fine-tune embeddings on domain-specific datasets</li>
        <li>✅ Implement bias mitigation techniques for fairer word representations</li>
        <li>✅ Train embeddings on larger and more diverse datasets</li>
    </ul>

    <h2>Installation</h2>
    <pre><code>git clone https://github.com/yourusername/word-embeddings.git
cd word-embeddings
pip install -r requirements.txt</code></pre>

    <h2>Usage</h2>
    <h3>Run Word2Vec Training</h3>
    <pre><code>python src/word2vec.py</code></pre>
    
    <h3>Train Neural Network-Based Embeddings</h3>
    <pre><code>python src/neural_embeddings.py</code></pre>
    
    <h3>Analyze Pretrained Embeddings</h3>
    <pre><code>jupyter notebook notebooks/pretrained_glove.ipynb</code></pre>

    <h2>Dependencies</h2>
    <ul>
        <li>Python 3.8+</li>
        <li>NumPy</li>
        <li>Matplotlib</li>
        <li>NLTK</li>
        <li>Scikit-learn</li>
        <li>TensorFlow/Keras</li>
        <li>PyTorch</li>
        <li>Gensim</li>
        <li>Jupyter Notebook</li>
    </ul>

    <h2>Contributors</h2>
    <p><strong>Your Name</strong> - <a href="https://github.com/yourusername">GitHub Profile</a></p>

    <h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

    <h2>Acknowledgments</h2>
    <ul>
        <li>"Speech and Language Processing" by Jurafsky & Martin</li>
        <li>"Efficient Estimation of Word Representations in Vector Space" - Mikolov et al.</li>
        <li>Jay Alammar's Blog on Word2Vec & Word Embeddings</li>
    </ul>
</body>
</html>

