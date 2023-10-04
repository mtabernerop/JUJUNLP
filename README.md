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

## Repository Structure

The repository is organized as follows:
- `dataset/`: contains the scripts that generate plots that ease the analysis of the training set structure.
- `task<1,2a,2b>/`: contains the scripts for training the transformers and predicting with the best resulting ensemble for `task<1,2a,2b>`.
- `graphs/`: contains the scripts that generate plots that ease the analysis of the results of a saved ensemble experiment.

## Experimentation

For the sake of completeness and in an attempt to improve the results obtained by the transformer assemblers, each run was repeated a total of 6 times with the different combinations of the following hyperparameters:

| Hyperparameter | Values |
|:--------------:|:------:|
| *Optimizer* | `{AdamW, Adafactor}` |
| *Learning rate* | `{2e-05, 4e-05, 8e-05}` |

The results of this experimentation can be found at [this link](https://drive.google.com/drive/folders/164RbDq_ndTWPPKahaO5LKsXRYEYowTem?usp=sharing).

## Results

The following table records the official results of the HUHU@IberLEF 2023 shared task. The metrics recorded by the best (winning) approach in each task and the best performing baseline are indicated alongside the name of the system that registered them. For the two runs submitted of our system (JUJUNLP<sub>1</sub> and JUJUNLP<sub>2</sub>, respectively), the position achieved in the final ranking is shown in parentheses. The metrics are the F1-score, weighted F1-score and RMSE in subtasks 1, 2A and 2B, respectively.

| System | Subtask 1 | Subtask 2A | Subtask 2B |
|:------:|:---------:|:----------:|:----------:|
| Best approach | 0.820 (RETUYT-INCO<sub>1</sub>) | 0.796 (JUJUNLP<sub>1</sub>) | 0.855 (M&C<sub>2</sub>) |
| JUJUNLP<sub>1</sub> | 0.772 (12) | 0.796 (1) | 0.934 (22) |
| JUJUNLP<sub>2</sub> | 0.722 (27) | 0.774 (4) | 0.939 (25) |
| Best baseline | 0.789 (BLOOM_1B1) | 0.760 (BETO) | 0.874 (BETO) |

## References

- HUHU 2023 Competition: https://sites.google.com/view/huhuatiberlef23/huhu
- Simple Transformers: https://simpletransformers.ai/
- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- Scikit-learn: https://scikit-learn.org/
- Seaborn: https://seaborn.pydata.org/
- Matplotlib: https://matplotlib.org/

## Citation

Please cite [our work]() if you use it.

```bib
@article{cruz2023unity,
  title={In unity, there is strength: On weighted voting ensembles for hurtful humour detection},
  author={Cruz, J and Elvira, L and Tabernero, M and Segura-Bedmar, I},
  journal={Iber-LEF@ SEPLN},
  year={2023}
}
```
