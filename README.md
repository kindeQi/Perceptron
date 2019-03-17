# Perceptron
- Spam mail detection
- You can download the dataset here: [Dataset](https://myleott.com/op_spam_v1.4.zip)
- Folder 5 for validation, folder1 - 4 for training
- Use tf-idf for each document
- Vanilla Perceptron Average F1: 0.9014
- Average Perceptron Average F1: 0.9015

# How to use
1. Train the Perceptron model, `python perceplearn3.py`
2. Trained 2 model: `averagedmodel.txt and vanillamodel.txt`
3. Validate the model: `python perpercepclassify3.py averagedmodel.txt` or `python perpercepclassify3.py vanillamodel.txt`
