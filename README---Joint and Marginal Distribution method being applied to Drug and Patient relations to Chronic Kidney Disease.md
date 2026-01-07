Applying Joint and Marginal Distribution to Drug and Patient relationship to Chronic Kidney Disease Predictions, 
I learned Joint and Marginal Distribution while taking Data Mining for my data science degree.

import pandas as pd
import kagglehub
import os

# ---------------------------------------------------------
# Make Sure you download dataset from Kaggle
# ---------------------------------------------------------
path = kagglehub.dataset_download("ziya07/drugpatient-dataset-for-ckd-prediction")
print("Path to dataset files:", path)

# This dataset Should contains a CSV file I f it doesnt my apologies; Here goes nothing..
#P.S---- (You may need to print os.listdir(path) to confirm the filename.)
file_list = os.listdir(path)
print("Files in dataset:", file_list)

#  You need to load the correct CSV file (assuming THIS CODE WONT ALLOW YOU TO EXECUTE. I would shoot for Google Colab since I was able to run successfully but also i could link the path to the Kaggle Page where the Data Set was located, or has a specific name)
# Make sure you check the file_list to ensure you're picking the right one.
csv_path = os.path.join(path, 'CKD_NephrotoxicDrug_Dataset.csv') # Explicitly use the correct filename
# Alternatively, if you're certain it's always file_list[1]:
# csv_path = os.path.join(path, file_list[1])
df = pd.read_csv(csv_path)

print("\nDataset Loaded Successfully!")
print(df.head())

# ---------------------------------------------------------
# 2. Distribution Functions
# ---------------------------------------------------------
def calculate_joint_distribution(df, features, class_label):
    joint_dist = df.groupby(features + [class_label]).size().reset_index(name='Count')
    total_count = df.shape[0]
    joint_dist['Probability'] = joint_dist['Count'] / total_count
    return joint_dist

def calculate_marginal_distribution(df, feature, class_label):
    marginal_dist = df.groupby([feature, class_label]).size().reset_index(name='Count')
    total_count = df.shape[0]
    marginal_dist['Probability'] = marginal_dist['Count'] / total_count
    return marginal_dist

# ---------------------------------------------------------
# 3. Choose features from the CKD dataset
# ---------------------------------------------------------
# These example columns in the CKD dataset:
# ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']

# Let's start to compute distributions using:
# - 'hypertension' (Hypertension)
# - 'diabetes' (Diabetes Mellitus)
# - 'ckd_risk_label' (CKD or Not CKD)

features = ['hypertension', 'diabetes']
class_label = 'ckd_risk_label'

# ---------------------------------------------------------
# 4. Compute Joint Distribution
# ---------------------------------------------------------
joint_dist = calculate_joint_distribution(df, features, class_label)
print("\nJoint Distribution:")
print(joint_dist)

# ---------------------------------------------------------
# 5. Compute Marginal Distributions
# ---------------------------------------------------------
marginal_htn = calculate_marginal_distribution(df, 'hypertension', class_label)
print("\nMarginal Distribution of Hypertension (hypertension):")
print(marginal_htn)

marginal_dm = calculate_marginal_distribution(df, 'diabetes', class_label)
print("\nMarginal Distribution of Diabetes (diabetes):")
print(marginal_dm)
