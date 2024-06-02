# DormantCustomerReignition

Objectives

User behavior analysis is essential in modern business, helping us understand reactions to products or services and forming a basis for marketing strategies. Analyzing the relationship between time and customer purchases reveals purchase volumes, allowing us to develop strategies to boost sales using coupons.

Key Steps:

Time Slot Analysis:

Aggregate hourly customer purchases to create time-series data.
Visualize data to identify periods of increasing and decreasing sales.
Distribute coupons strategically before active periods and during inactive ones to boost sales.
Dormant Customer Strategies:

Clustering:

Classify dormant customers into two groups: low-purchase and high-value dormant customers.

Low-Purchase Customers:

Minimal purchase frequency and amount due to budget constraints, dissatisfaction, or decision-making difficulties.

Strategies: personalized recommendations, VIP benefits, free shipping/returns, product reviews, improved customer service, and secure transactions.

High-Value Dormant Customers:

Previously significant purchases but currently inactive due to one-time purchases, shifting interests, or competitor products.

Strategies: personalized recommendations, repeat purchase discounts, special promotions, premium services, product upgrades, and membership programs.

Goal:
Using data on customer responses to product alerts and coupons, we can personalize approaches for each group to activate dormant customers and prevent churn.


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
