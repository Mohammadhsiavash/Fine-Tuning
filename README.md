# Sentiment Analysis Using BERT

## Overview
This project implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers). The model is fine-tuned on the SST-2 dataset from the GLUE benchmark to classify text as either positive or negative.

## Features
- **Preprocessing & Tokenization**: Uses Hugging Face's `AutoTokenizer` to preprocess text data.
- **Model Architecture**: Implements `BertForSequenceClassification` from the `transformers` library.
- **Training & Fine-Tuning**: Utilizes `TrainingArguments` for optimized training.
- **Evaluation & Metrics**: Uses the `evaluate` library for performance analysis.
- **Model Saving**: Saves trained models for inference and reuse.

## Dataset
The project uses the **SST-2 dataset** from the GLUE benchmark, which consists of sentences labeled as either positive or negative.

## Installation
Ensure you have Python installed, then install the necessary dependencies:
```bash
pip install torch transformers datasets evaluate
```

## Usage
### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/sentiment-analysis-bert.git
cd sentiment-analysis-bert
```

### 2. Run the Training Script
```bash
python train.py
```

### 3. Evaluate the Model
```bash
python evaluate.py
```

### 4. Use the Model for Inference
```python
from transformers import AutoTokenizer, BertForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("path_to_saved_model")

text = "I love this movie!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print("Sentiment:", "Positive" if predictions.item() == 1 else "Negative")
```

## Results
The model is evaluated using accuracy and F1-score. Fine-tuning BERT achieves high performance on SST-2, demonstrating its effectiveness in sentiment classification.

## Future Improvements
- Experiment with different transformer architectures (e.g., RoBERTa, DistilBERT).
- Optimize hyperparameters for better accuracy.
- Extend the model to multi-class sentiment classification.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SST-2 Dataset](https://gluebenchmark.com/tasks)
- [PyTorch](https://pytorch.org/)

