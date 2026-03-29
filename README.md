# CodeAlpha_Iris_Flower_Classification
CodeAlpha Internship Project - Iris Flower Classification using Machine Learning
# Iris Flower Classification

This project is developed as part of CodeAlpha Internship.

## Description
This project uses Machine Learning to classify Iris flowers into:
- Setosa
- Versicolor
- Virginica

based on flower measurements.

## Features
- Data preprocessing
- Model training using Scikit-learn
- Accuracy evaluation
- Data visualization

## How to Run
1. Install required libraries
2. Run the Python file
3. View predictions and graphs

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
## output
First 5 rows:
    Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa

Columns:
 Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species'],
      dtype='str')

Accuracy: 100.00%

Classification Report:
                  precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00         9
 Iris-virginica       1.00      1.00      1.00        11

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30


--- Test with Manual Input ---
Enter Sepal Length: 5.1
Enter Sepal Width: 3.5
Enter Petal Length: 1.4
Enter Petal Width: 0.2

Predicted Flower: Iris-setosa
