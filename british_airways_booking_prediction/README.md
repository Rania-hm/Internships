\# ✈ British Airways – Customer Booking Prediction



\## Project Overview

This project was completed as part of a British Airways virtual internship.

The objective is to predict whether a customer will complete a booking

and identify the key factors influencing booking behavior.



\## Business Objective

\- Predict booking completion

\- Understand customer characteristics driving bookings

\- Provide insights to support marketing and pricing strategies



\## Dataset

\- Source: British Airways customer booking data

\- Target variable: `booking\_complete` (0 = No booking, 1 = Booking)

\- Dataset includes customer demographics, travel details, and booking behavior



\## Methodology

1\. Exploratory Data Analysis (EDA)

2\. Data cleaning and preprocessing

3\. Handling class imbalance using SMOTE

4\. Training a Random Forest classifier

5\. Threshold optimization using F1 score

6\. Feature importance analysis

7\. Business insight interpretation



\## Model Used

\- Random Forest Classifier

\- Tuned hyperparameters:

&nbsp; - n\_estimators = 800

&nbsp; - max\_depth = 30

&nbsp; - min\_samples\_split = 4

&nbsp; - min\_samples\_leaf = 3

&nbsp; - max\_features = 0.5

&nbsp; - class\_weight = {0:1, 1:2}



\## Model Performance

\- Accuracy: 72.1%

\- Recall (Booking class): 70.5%

\- F1 Score (Booking class): 0.431



Confusion Matrix:

\[\[6158 2346]

\[ 441 1055]]

## Key Predictive Features
- Flight duration
- Booking origin (Malaysia, Australia, Indonesia, China)
- Purchase lead time
- Length of stay
- Flight hour
- Number of passengers

## Business Insights
- Customers from specific regions show higher booking probability
- Shorter purchase lead time strongly correlates with booking completion
- Model is suitable for identifying high-intent customers for marketing campaigns

## Repository Structure
- `data/` – dataset
- `src/` – production-ready Python script
- `results/` – plots and metrics
- `presentation/` – final business summary slide

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

## Author
Ania
