# DormantCustomerReignition

Architecture

1. Data Loading and Initial Analysis

Load the data from a CSV file.
Display basic information and statistics about the dataset.
Extract date-related features from the event timestamps.

2. Data Exploration and Visualization

Visualize event counts on an hourly and monthly basis.
Visualize the distribution of different event types.

3. Data Preparation

Extract the latest event for each user.
Calculate the number of days since the last event for each user.
Calculate the event count and total purchase amount for each user.
Fill missing values in categorical columns using KNN.
Encode categorical columns using label encoding.

4. Data Scaling

Apply different scaling techniques (StandardScaler, MinMaxScaler, RobustScaler).

5. Clustering Analysis

Determine the optimal number of clusters using the Elbow Method.
Apply KMeans clustering and visualize the results.
Apply Hierarchical Agglomerative Clustering and visualize the dendrogram.
Evaluate clustering performance using silhouette scores.

6. Main Execution

Integrate all steps in a structured workflow.
Identify the best clustering configuration.
