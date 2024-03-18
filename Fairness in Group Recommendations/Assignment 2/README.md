# Fairness in Group Recommendations - Assignment 2

This Directory contains the Code related to the Second Assignment for the 'Fairness in Group Recommendations' lectures, held by Prof. Kostantinos Stefanidis.

| File | Description |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [experiments.ipynb](https://github.com/DecioXXIV/ATCS-Assignments/blob/main/Fairness%20in%20Group%20Recommendations/Assignment%202/experiments.ipynb) | Jupyter Notebook, it shows the execution of the "User-Based Collaborative Filtering" approach for Group Recommandations |
| [similarity.py](https://github.com/DecioXXIV/ATCS-Assignments/blob/main/Fairness%20in%20Group%20Recommendations/Assignment%202/similarity.py) | Python file, it contains the implementation of the Pearson Similarity Function |
| [predict.py](https://github.com/DecioXXIV/ATCS-Assignments/blob/main/Fairness%20in%20Group%20Recommendations/Assignment%202/predict.py) | Python file, it contains the implementation of the Prediction Function, which exploits the Similarity between Users to recommend a "new" Item to a given User |
| [group.py](https://github.com/DecioXXIV/ATCS-Assignments/blob/main/Fairness%20in%20Group%20Recommendations/Assignment%202/group.py) | Python file, it contains the implementations of the Aggregation Functions (Average, Least Misery, Weighted Disagreements Sum) for the predicted Scores and the 'Group Recommendation' Function |
| [evaluate.py](https://github.com/DecioXXIV/ATCS-Assignments/blob/main/Fairness%20in%20Group%20Recommendations/Assignment%202/evaluate.py) | Python file, it contains the implementation of a function that computes the Kendall Tau Distance between two rankings |