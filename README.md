# machine_learning_project-unsupervised-learning

In this Project, we are going to perform a full unsupervised learning machine learning project on a "Wholesale Data" dataset. The dataset refers to clients of a wholesale distributor. It includes the annual spending in monetary units (m.u.) on diverse product categories. The data source of this dataset is from [Kaggle](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set).

The dataset is originally from the [UCI Wholesale customers Data Set](https://archive.ics.uci.edu/dataset/292/wholesale+customers). However, the categorical variables has been encoded to numerical values in the [Kaggle Link](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set).

**Project objective**: To perform unsupervised learning techniques to find patterns in the product selling, the customers spending behavior from a wholesale distributor.

## Data Information

The dataset is originally from the [UCI Wholesale customers Data Set](https://archive.ics.uci.edu/dataset/292/wholesale+customers). However, the categorical variables has been encoded to numerical values in the [Kaggle Link](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set).

Additional Information:

| Variable            | Type          | Description                                                           | 
|---------------------|---------------|-----------------------------------------------------------------------| 
| Channel             | Nominal       | Customers' Channel - 1: Horeca (Hotel/Restaurant/Café) or 2: Retail channel |
| Region              | Nominal       | Customers' Region - 1: Lisbon, 2: Oporto, or 3: Other Region          |
| Fresh               | Continuous    | Annual spending (m.u.) on fresh products                              | 
| Milk                | Continuous    | Annual spending (m.u.) on milk products                               |
| Grocery             | Continuous    | Annual spending (m.u.) on grocery products                            | 
| Frozen              | Continuous    | Annual spending (m.u.) on frozen products                             | 
| Detergents_Paper    | Continuous    | Annual spending (m.u.) on detergents and paper products               | 
| Delicassen          | Continuous    | Annual spending (m.u.) on delicatessen products                       |

## Project Outcomes
- Unsupervised Learning: perform unsupervised learning techniques on a wholesale data dataset. The project involves four main parts: exploratory data analysis and pre-processing, KMeans clustering, hierarchical clustering, and PCA.

### Project Description:
In this project, we will apply unsupervised learning techniques to a real-world data set and use data visualization tools to communicate the insights gained from the analysis.

The data set for this project is the "Wholesale Data" dataset containing information about various products sold by a grocery store.
The project will involve the following tasks:

-	**Exploratory data analysis and pre-processing**: We will import and clean the data sets, analyze and visualize the relationships between the different variables, handle missing values and outliers, and perform feature engineering as needed.
-	**Unsupervised learning**: We will use the Wholesale Data dataset to perform **k-means clustering, hierarchical clustering, and principal component analysis (PCA)** to identify patterns and group similar data points together. We will determine the optimal number of clusters and communicate the insights gained through data visualization.

The ultimate goal of the project is to gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked."

## Actual Project Flow:

### Part I : EExploratory data analysis and pre-processing

Include these steps:

1. Data Import
2. Data Description
3. Data Cleaning
4. Data Visualization & Exporation: Answers those questions by visualization: 1) Are there more Horeca or more Retail customers for this wholesaler?, 2) How are the customers of this wholesaler distributed across the regions?, 3) Are there any difefrences between the spending patterns from Horeca or more Retail customers across the product categories?
5. Outlier Dectection & Handling
6. Correlation Analysis
7. Feature Selection
8. Data Transformation

### Part II : KMeans Clustering

Include these steps:

1. Use the Elbow method for finding the optimal number of clusters in K-Means on X (with full features) and X_selected (with selected features) dataset
2. Train the model on X (the full features data set) with the optimal number of clusters
3. Train the model on X_selected (the selected features data set) with the optimal number of clusters
4. Compare the predictions of the 2 K-means clustering models using the full features and some selected features
5. Model conclusion

### Part III : Hierachical Clustering

Include these steps:

1. Draw the dendrogram to help decide the number of clusters for X (the full features) data set
2. Training the model on X (the full features) with the chosen number of clusters from the dendrogram
3. Draw the dendrogram to help decide the number of clusters for X_selected (the selected features data set)
4. Train the model on X_selected (the selected features data set) with the chosen number of clusters from the dendrogram
5. Compare the prediction cluster labels of the 2 Hierarchical clustering models using the full features and some selected features
6. Model conclusion

### Part IV : PCA

Include these steps:

1. Apply PCA the first time, keeping all components equal to the original number of dimensions and see how well PCA captures the variance of our data
2. Run PCA again with the chosen number components from the first run
3. Model conclusion


### Part V : Conclusion

Include:

- Conclusion from the exploratory data analysis (EDA) conducted
- Conclusion from the the machine learning models developed

## Final Findings:

### Conclusion from the exploratory data analysis (EDA) conducted:

* Channel wise, the majority of the wholesaler's clients belong to the Horeca (Hotel/Restaurant/Café) channel, and significantly less common in Tthe Retail channel. Around 68% of the customers of this wholesaler are from Horeca places, whereas only 32% are Retail customers.

* Region wise, the biggest chunk of clients of the wholesaler are aggregated into the massive 'Other Region' category which doesn't explain well statiscally when compared with aggregation. 72% the customer base of this wholesaler are densely located there. No further details about which cities/locations contribute to 'Other Region', while 'Lisnon', 'Oporto' seems to be the largest cities/regions respectively in our studied country. Removing 'Other Region', Lisnon has far more clients than in  Oporto. Horeca customers are more present in Lisnon than in Operto.

* The wholesaler should consider re-categorizing their client region segmentation to separate more significant cities/regions out of 'Other Region'. This will positively allow better insights into the regions and channels mapping, instead of having a big chunk of density amalgamation in 'Other Region' for both Region and Channel.

* Product Category wise, there are 5 product categories with their annual spending per customer across: 'Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen'. The distributions of those categories reveals that the annual spending of those the majority of customers in our dataset have low to moderate spending. There are also a few outliers in such columns which represents a few clients who spend a lot. 

* In terms of spending patterns, nearly 68% of customers are from Horeca, their average annual spend for all categories is just $26,844 (assuming the spend is in dollars). Retail customers accounts for only 32% of the total number of customers, yet their average yearly spend is $46,619. Retail customers tend to spend more on Milk, Groceries and Detergents_paper than those fromHoreca customers. Both channels see similar overall spending patterns for the rest of the categories, despite Horeca customers slightly spend more on Fresh produce and Frozen products.

### Conclusion from the the machine learning models developed:

* Our goal of this project is to perform unsupervised learning techniques to find patterns in the product selling, the customers spending behavior from a wholesale distributor. The end goal might be trying to increase spending based on the hidden patterns in our dataset.  We tried KMeans clustering, Hierarchical clustering, and PCA algorithms. We also tried to compare the impacts of these models on 2 subsets of our data in the K-means and Hierarchical clustering models - one with all attributes and one with a selected combination of attributes. Due to PCA being so powerful in dimensional reduction and feature selection, the algorithm doesn't require repeating the algorithm to different subsets of the data.

* K-means neighboring are able to segment the clients data into an optimal number of 6 clusters of clients. There isn't any difference in the customers cluster groups, even when we removed the 'Channel' characteristic and the annual spending of the 'Detergents_Paper' and 'Milk' product categories. This might be an indicator of a modest quality of segmented customer groups.

* Hierarchical clustering provides 2 clusters of customers based on the categorical and spending inputs. There are variances in the customers cluster groups if we try 2 different combinations of the attributes, which has only 20 clients being listed matchingly in the 2 trials. This clustering algorithm shows a "better" response to changing up the customer segmentation subsets compared to K-means in this case.

* PCA is more powerful in the underlying structure of the wholesale customer data and finding the compound combinations of features best describe customers. It transformed the original data we have from having 8 columns down to only 2 top components that retains 92.80% of the original information. The top best combination include ['Fresh', 'Grocery', 'Detergents_Paper', 'Milk', 'Frozen', 'Delicassen', 'Channel', 'Region']; second b['Grocery', 'Milk', 'Detergents_Paper', 'Fresh', 'Frozen', 'Delicassen', 'Channel', 'Region']. However, the transformed data by PCA can be hard to intepret and communicate to business users, as it's inexplicitly hard to explain on why we choose those combination in layman's terms.

## Challenges

* The need to leverage a comprehensive knowledge and applicability of using of Python, Statistics/Math and Unsupervised Machine Learning concepts 
* Time constraints and increased work in data cleaning, EDA, pre-processing and model deployment for 2 sets of features (with full features and with selected features) for 3 unsupervised learning models

## Future Goals

* To expand the EDA and data prepprocessing stage to uncover patterns and insights that we have missed in the scope of this project, probably some bivariate and multivariate analysis between the categorical variables and numeric variables of spend.
* To test out a few more data transformation techniques and machine learning models that work with hightly skewed distributions and outliers to improve the output of our models.
* Maybe test out turning the unsupervised learning problem into a supervised one using regression or classification to predict the annual spend of each customer group.

