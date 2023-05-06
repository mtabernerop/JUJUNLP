# Weighted Voting Ensembles for Hurtful Humour Detection

This GitHub repository contains the code and resources used by the **JUJUNLP** group to participate in the **HUrtful HUmour (HUHU)** task at **IberLEF 2023**. The goal is to detect hurtful humour spreading prejudice in Twitter posts using Natural Language Processing (NLP) techniques.

## Tasks

The HUHU 2023 competition involved a total of four different tasks. These are listed below:
- *Task1 - HUrtful HUmour Detection*: binary classification task where it must be determined whether a prejudicial tweet is intended to cause humour or not.
- *Task2a - Prejudice Target Detection*: multi-label classification task where a total of four minority target groups must be identified on each tweet. 
- *Task2b - Degree of Prejudice Prediction*: regression task where it must be determined on a continuous scale from 1 to 5 how prejudicial the message is on average among minority groups.

## Methodology

The pursued approach involves using a weighted voting system of ensembles composed of different transformers. We trained several state-of-the-art NLP models on a large dataset of annotated tweets to create ensembles of classifiers with different architectures and configurations. We then combined the predictions of these ensembles using a weighted voting system to produce the final predictions.

We have used the following transformers for the ensembles:
- Multilingual BERT (cased, uncased)
- RoBERTa
- BETO (cased, uncased)
- DistilBERT

For each instance, the final classification decision is based on the weighted sum of outputs of these models. The novel weighted-voting system presented involves using each (normalized) transformer's metric score in the ensemble (F1-score or RMSE, depending on the task) to assess the importance of these in the final outputs of the ensemble (as opposed to the arithmetic mean typically used in conventional voting systems).

## Results

TO-DO

## References

- HUHU 2023 Competition: https://sites.google.com/view/huhuatiberlef23/huhu
- Simple Transformers: https://simpletransformers.ai/
- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- Scikit-learn: https://scikit-learn.org/
- Seaborn: https://seaborn.pydata.org/
- Matplotlib: https://matplotlib.org/
