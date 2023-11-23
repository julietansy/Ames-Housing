# **Ames Housing**

## Introduction

The Ames Housing dataset encompasses a detailed record of the sale of individual residential property spanning from 2006 to 2010, featuring close to 2930 entries for properties located in Ames, Iowa. With 81 distinct variables, it offers comprehensive information on a multitude of property characteristics. This dataset offers an expansive view of features crucial in evaluating a house, presenting thorough details on nearly every aspect of these properties.

## Objective

Investigate the correlation of features on property against sales price.

## Python Libraries

        # Import required libraries
        import pandas as pd
        import numpy as np
        import scipy.stats as stats
        import seaborn as sns
        from sklearn.preprocessing import LabelEncoder
        import matplotlib.pyplot as plt
        from scipy import stats



## Data Source

The dataset originates from the [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset/code) on Kaggle. The variables in this dataset focus on both the quality and quantity of numerous physical attributes of the properties.

        # Load raw data
        data = pd.read_csv('../input/ames-housing-dataset/AmesHousing.csv')
        
    
There are 39 columns which are numerical and 43 columns are categorical.

2 out of 39 numerical variables shall be dropped from the dataset as they does not provide valuable information. They are Order and PID, which are the index and unique transaction number, respectively.

Of the remaining variables, 12 are discrete and 25 are continuous. Discrete variables mainly cover rankings or quantities of certain attributes, such as overall quality ranked from 1 to 5 or the number of bedrooms in the property. Continuous variables encompass descriptive features, with several ranked qualitative descriptions that can be transformed into ordinal variables for easier reference in subsequent analysis.   

        
        # Replace spacing in column names with underscores
        data.columns = data.columns.str.replace(' ', '_')
        
        # Print the shape of the dataset
        print("This dataset contains ", data.shape[0], " rows and ", data.shape[1], " columns")


    This dataset contains  2930  rows and  82  columns


        # Define variables which are numerical 
        numeric_var = data.select_dtypes('number')
        
        # Explore numerical variables
        print(numeric_var.describe(include = 'all'))
        print(numeric_var.info()) 
  
        # Define variables which are categorical
        categorical_var = data.select_dtypes(exclude='number')
        
        # Explore categorical variables
        print(categorical_var.describe(include = 'all'))


## Data Cleaning

The data cleaning process focuses primarily on handling missing values by replacing them with suitable placeholders based on the data types. Null values are substituted with '0' for numerical columns and 'None' for categorical columns. Additionally, a check for duplicated rows was conducted, ensuring data integrity throughout the dataset.

The detailed breakdown of the data cleaning process can be found in the appendix at the end of this document.


## Data Processing

The data processing phase involved the creation of new features and the encoding of categorical variables to numerical representations using label encoding.

### Feature Engineering:
Sold House Age Calculation:
The 'Sold_House_Age' variable was derived by subtracting the 'Year_Built' from the 'Yr_Sold', providing insights into the age of the property at the time of sale.

        # Create the "Sold_House_Age" variable
        data_final["Sold_House_Age"] = data_final["Yr_Sold"] - data_final["Year_Built"]


### Label Encoding:
Several categorical variables were encoded using label encoding to transform qualitative descriptions into numerical representations. The mapping of categories to numerical values is as follows:

**MSSubClass:** Encoding specific subclasses into more concise representations ('SC20', 'SC30', etc.).

**Other Categorical Variables:** Replaced categorical values with corresponding numerical labels according to specific mappings provided, ensuring a numerical format for these variables in the dataset.


        # Label encoding of variable
        encode = {"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                               50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                               80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                               150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                  "Street" : {"Grvl" : 1, "Pave" : 2},
                  "Alley" : {"None" : 0, "Grvl" : 1, "Pave" : 2},
                  "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                  "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4},
                  "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                  "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                  "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                  "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                  "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                  "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                  "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                    "ALQ" : 5, "GLQ" : 6},
                  "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                    "ALQ" : 5, "GLQ" : 6},
                  "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                  "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                  "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                  "Min2" : 6, "Min1" : 7, "Typ" : 8},
                  "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                  "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                  "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                  "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                  "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                  "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                              7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                              }
        
        data_final = data_final.replace(encode)


## EDA and Visualization on Numerical Variables

### Distribution of Sales Price
The histogram for Distribution of Sales Price reveals it to be right-skewed which can also be intepreted from the median being lower than the mean. The QQ plot also reflects a non-linear pattern, suggesting that the tails of the distribution had extement values (outliers) and are positively skewed, as compared to a normal distribution. Furthermore, the calculated skewness value of 1.74 and kurtosis of 5.11 further supports the interpretation of a right-skewed distribution. (The acceptable range for skewness are between -0.5 and 0.5 and between -2 and 2 for kurtosis.)

The right-skewed distribution represents that the majority of the properties are of a lower sale prices, while there are a few housing properties with significantly higher sale prices. This is evident from the longer right tail in the plot. The presence of outliers on the higher end of sales prices shall be evaluated as there will be a substantial impact on the increased mean.

Although the results of the Shapiro-Wilk test statistic of 0.88 and a p-value of 0.00 provides a strong evidence to reject the null hypothesis of a normal distribution, it is important to consider that the Shapiro-Wilk test assumes that the dataset does not have significant skewness or kurtosis and does not have outliers. Therefore, we shall hold on to the Shapiro-Wilk test until determining the presence of outliers.

In order to gain a comprehensive understanding of the Sales Price distribution, it is recommended to consider additional exploratory data analysis techniques. These may exploring the correlation between the features and sales price.

![download](https://github.com/julietansy/Ames-Housing/assets/151416878/4d5fde34-6fb4-47b8-b369-42e48ad8594f)

        # Extract the "Sale Price" column from the data
        sales_price = data_final["SalePrice"]
        
        # Plot the distribution of "Sales Price"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(sales_price, bins=30, edgecolor='black')
        ax1.set_xlabel("Sales Price")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Sales Price")
        
        # Add a vertical line representing the mean or the median
        mean = sales_price.mean()
        median = sales_price.median()
        ax1.axvline(mean, color='red', linestyle='--', label='Mean')
        ax1.axvline(median, color='green', linestyle='--', label='Median')
        ax1.legend()
        
        # QQ plot
        sm.qqplot(sales_price, dist=stats.norm, line='s', ax=ax2)
        ax2.set_title("QQ Plot of Sales Price")
        
        # Plot graph
        plt.show()

        # Calculate the Skewness, Kurtosis and Shapiro Test of SalePrice
        saleprice_skewness = stats.skew(data_final['SalePrice'])
        saleprice_kurtosis = stats.kurtosis(data_final['SalePrice'])
        shapiro_test_stat, shapiro_test_pvalue = stats.shapiro(data_final['SalePrice'])
        
        # Print results
        print("Skewness: {:.2f}" .format(saleprice_skewness))
        print("Kurtosis: {:.2f}" .format(saleprice_kurtosis))
        print("Shapiro Test Statistic: {:.2f}" .format(shapiro_test_stat))
        print("Shapiro Test p-value: {:.2f}" .format(shapiro_test_pvalue))


### Identification of Outliers in Sales Price

The IQR is calculated as 84,000, with upper bound of 339,500 and lower bound of 3,500. Any Sales Price falling above the upper bound and below the lower bound are considered as potential outliers.

There are 137 (4.68%) potential outliers which are derived from information obtained from lower and upper bound.



![download](https://github.com/julietansy/Ames-Housing/assets/151416878/4a802319-abff-40e1-a61f-5beb4f2e1c5b)


        # Create a box plot to visualize the distribution and identify outliers
        fig = plt.figure(figsize=(10, 5))
        box_plot = plt.boxplot(sales_price)
        outliers = box_plot["fliers"][0].get_ydata()
        outlier_count = len(outliers)
        
        # Print the number of outliers
        print("Number of outliers:", outlier_count)
        
        # Display the box plot with the outlier count
        plt.boxplot(sales_price)
        plt.xlabel("Sales Price")
        plt.ylabel("Value")
        plt.title("Box Plot - Sales Price\nOutliers: {}".format(outlier_count))
        plt.show()
        
        # Calculate the IQR
        Q1 = sales_price.quantile(0.25)
        Q3 = sales_price.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers
        upper_bound = Q3 + (1.5 * IQR)
        lower_bound = Q1 - (1.5 * IQR)
        print('IQR: ', IQR, '\nUpper Bound: ', upper_bound, '\nLower Bound: ', lower_bound, '\n')
        
        # Find the outliers
        outliers = sales_price[(sales_price < lower_bound) | (sales_price > upper_bound)]
        
        # Print the outliers
        print("Outliers in SalePrice:")
        print(outliers)

        # Convert outliers to a DataFrame
        outliers_df = pd.DataFrame(outliers, columns=data_final.columns)
        
        # Get the indices of the outliers
        outlier_indices = outliers_df.index
        
        # Retrieve the corresponding neighborhoods and sales prices of the outliers
        outliers_data = data_final.loc[outlier_indices, ['Neighborhood', 'SalePrice']]
        
        # Group the outliers by neighborhood and calculate the count and mean sales price
        outliers_grouped = outliers_data.groupby('Neighborhood')['SalePrice'].agg(['count', 'mean'])
        outliers_grouped['mean'] = outliers_grouped['mean'].round(2)
        
        # Print the grouped outliers with count and mean sales price
        print("Outliers grouped by neighborhood with count and mean sales price:")
        print(outliers_grouped)
        
        # Calculate the percentage of outliers
        outlier_percentage = (len(outlier_indices) / len(data_final)) * 100
        
        # Print the percentage of outliers
        print("\nPercentage of outliers in the dataset: {:.2f}%".format(outlier_percentage))

### Identify Features Correlated with Sales Price

Nonetheless, the sales of Ames Housing were sold over the period of 2006 to 2010 and had multiple variables which would affect the range of sales. Working on domain knowledge on housing sales prices, the following are variables which shall be explored further to determine if it is a predictor of sales price.

1. Overall Quality
2. Sold House Age (Direct relation to Year Build)
3. Living Area
   - First Floor Living Area
   - Ground Living Area
   - Total Basement Area
4. Garage Cars (Garage Area will determine how many cars can be parked in the garage)
5. Full Bath


![download](https://github.com/julietansy/Ames-Housing/assets/151416878/ca81fafb-a650-47bd-a8a3-3b4ac8746673)

        # Identify the top 10 features with highest absolute correlation with Sales Price
        correlation_matrix = data_final.corr().abs()
        top_10_features = correlation_matrix.nlargest(10, 'SalePrice')['SalePrice'].index
        
        # Create a subset of data with Sales Price and top 10 features
        subset_data = data_final[top_10_features]
        
        # Calculate the correlation matrix
        corr_matrix = subset_data.corr()
        
        # Plot the correlation matrix using a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
        plt.title('Correlation Matrix of Top 10 Features against Sales Price')
        plt.show()


**1. Overall Quality**

Upon analyzing the data, it is evident that the overall quality of the house has a strong positive correlation of 0.8 with the sales price. This is supported by the bar chart where the sales price increases with an increase in overall quality. Interpreting from the box plot, when the house is of a low overall quality, the sales price range are at the lower range and have a narrower range of sales prices, indicating less variablity. As the overall quality increases, it widens the sales price range and outliers are perceived. These outliers represent houses with higher sales prices compared to the majority of houses within their respective overall quality categories. This suggests that there are certain exceptional houses with higher sales prices despite having the same overall quality as other houses. These exceptional houses may be influenced by various features, contributing to their higher market value.

![download](https://github.com/julietansy/Ames-Housing/assets/151416878/d89c2f4c-93ba-4f65-8691-1bbc72f46cf2)


        # Calculate the mean sales price by Overall_Qual
        mean_sales_price_by_overall_qual = data_final.groupby("Overall_Qual")["SalePrice"].mean().reset_index().round(2)
        
        # Sort the data by Overall_Qual in ascending order
        mean_sales_price_by_overall_qual = mean_sales_price_by_overall_qual.sort_values("Overall_Qual")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the mean sales price by Overall_Qual
        ax1.bar(mean_sales_price_by_overall_qual["Overall_Qual"], mean_sales_price_by_overall_qual["SalePrice"])
        ax1.set_xlabel("Overall Quality")
        ax1.set_ylabel("Mean Sales Price")
        ax1.set_title("Mean Sales Price by Overall Quality")
        
        # Create a box plot of Sales Price by Overall_Qual
        sns.boxplot(x=data_final["Overall_Qual"], y=data_final["SalePrice"], ax=ax2)
        ax2.set_xlabel("Overall Quality")
        ax2.set_ylabel("Sales Price")
        ax2.set_title("Sales Price Distribution by Overall Quality")
        
        # Adjust the spacing between subplots
        fig.tight_layout()
        
        # Show the plots
        plt.show()
        
        # Calculate the correlation between sales price and Overall_Qual
        correlation = data_final['SalePrice'].corr(data_final['Overall_Qual'])
        
        # Print the correlation
        print("Correlation between Sales Price and Overall Quality: {:.2f}".format(correlation))


**2. Sold House Age**

The sold house age exhibits a negative correlation of 0.56, implying that the mean sales prices decreases with an increase in the age of the house at the point of sale. Analyzing the box plot, it is observed that the difference in mean sales prices between different housing age groups is more pronounced for houses that are relatively young. As the age of the house increases, the difference in mean sales prices becomes less fluctuating, indicating a relatively stable pricing pattern for houses older than 30 years.

Nonetheless, the higher number of outliers in the box plot reveals that there might be other features which are influential, beyond the age of the house on sales prices.


![download](https://github.com/julietansy/Ames-Housing/assets/151416878/42b4ec0f-80a4-4e72-b716-ca742b7efd28)

        # Define the bin edges for grouping sold age
        bin_edges = [0, 10, 20, 30, 40, 50, float('inf')]
        bin_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50+']
        
        # Create a new column "Sold_House_Age_Group" with the corresponding age bin labels
        data_final['Sold_House_Age_Group'] = pd.cut(data_final['Sold_House_Age'], bins=bin_edges, labels=bin_labels)
        
        # Calculate the mean sales price by Sold_House_Age group
        mean_sales_price_by_soldage_group = data_final.groupby("Sold_House_Age_Group")["SalePrice"].mean().reset_index().round(2)
        
        # Sort the data by Sold_House_Age in ascending order
        mean_sales_price_by_soldage_group = mean_sales_price_by_soldage_group.sort_values("Sold_House_Age_Group")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the mean sales price by Sold_House_Age group
        ax1.bar(mean_sales_price_by_soldage_group["Sold_House_Age_Group"], mean_sales_price_by_soldage_group["SalePrice"])
        ax1.set_xlabel("Sold Age Group")
        ax1.set_ylabel("Mean Sales Price")
        ax1.set_title("Mean Sales Price by Sold Age Group")
        
        # Create a box plot of Sales Price by Sold Age Group
        sns.boxplot(x=data_final["Sold_House_Age_Group"], y=data_final["SalePrice"], ax=ax2)
        ax2.set_xlabel("Sold Age Group")
        ax2.set_ylabel("Sales Price")
        ax2.set_title("Sales Price Distribution by Sold Age Group")
        
        # Adjust the spacing between subplots
        fig.tight_layout()
        
        # Show the plots
        plt.show()
        
        # Calculate the correlation between sales price and Sold_House_Age
        correlation = data_final['SalePrice'].corr(data_final['Sold_House_Age'])
        
        # Print the correlation
        print("Correlation between Sales Price and Sold House Age: {:.2f}".format(correlation))


**3. Living Area**

The correlation coefficient between Sales Price and Total Living Area stands at an impressive 0.79. This indicates a strong positive relationship between the two variables. As the Total Living Area increases, there's a noticeable tendency for the Sales Price to rise, signifying that larger living spaces tend to be associated with higher property prices. 

The bar chart showcasing the mean Sales Price across different Total Living Area bins illustrates a consistent ascending trend with the increase in living area. The graph depicts a gradual increase in the mean Sales Price until it peaks within the 6000-7000 square feet range. However, beyond this point, there's a considerable decline in the mean Sales Price.

The box plot reveals intriguing insights into the distribution of Sales Price concerning Total Living Area. Particularly, there's a noteworthy presence of outliers up to the 4000 square feet mark. Beyond this threshold, the number of outliers decreases significantly, signifying a shift in the distribution of Sales Price as the living area surpasses the 4000 square feet mark.

![download](https://github.com/julietansy/Ames-Housing/assets/151416878/a23ef0ed-3aab-40b5-abfc-14744f53b7a8)

        # Calculate the total living area
        data_final['Gr_Liv_Area'] = pd.to_numeric(data_final['Gr_Liv_Area'], errors='coerce')
        data_final['Total_Bsmt_SF'] = pd.to_numeric(data_final['Total_Bsmt_SF'], errors='coerce')
        data_final['Total_Living_Area'] = data_final['Gr_Liv_Area'] + data_final['Total_Bsmt_SF']
        
        # Define the bin edges for grouping Total_Living_Area
        bin_edges = [data_final['Total_Living_Area'].min()-1, 2000, 3000, 4000, 5000, 6000, 7000, 8000, float('inf')]
        bin_labels = ['<2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000-7000', '7000-8000','8000+']
        
        # Create a new column "Total_Living_Area_Group" with the corresponding area bin labels
        data_final['Total_Living_Area_Group'] = pd.cut(data_final['Total_Living_Area'], bins=bin_edges, labels=bin_labels)
        
        # Calculate the mean sales price by Total_Living_Area group
        mean_sales_price_by_total_living_area_group = data_final.groupby("Total_Living_Area_Group")["SalePrice"].mean().reset_index().round(2)
        
        # Sort the data by Total_Living_Area in ascending order
        mean_sales_price_by_total_living_area_group = mean_sales_price_by_total_living_area_group.sort_values("Total_Living_Area_Group")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the mean sales price by Total_Living_Area group
        ax1.bar(mean_sales_price_by_total_living_area_group["Total_Living_Area_Group"], mean_sales_price_by_total_living_area_group["SalePrice"])
        ax1.set_xlabel("Total Living Area Group")
        ax1.set_ylabel("Mean Sales Price")
        ax1.set_title("Mean Sales Price by Total Living Area Group")
        
        # Create a box plot of Sales Price by Total Living Area Group
        sns.boxplot(x=data_final["Total_Living_Area_Group"], y=data_final["SalePrice"], ax=ax2)
        ax2.set_xlabel("Total Living Area Group")
        ax2.set_ylabel("Sales Price")
        ax2.set_title("Sales Price Distribution by Total Living Area Group")
        
        # Adjust the spacing between subplots
        fig.tight_layout()
        
        # Calculate the correlation between sales price and Total_Living_Area
        correlation = data_final['SalePrice'].corr(data_final['Total_Living_Area'])
        
        # Print the correlation
        print("Correlation between Sales Price and Total Living Area: {:.2f}".format(correlation))
        
        # Show the plots
        plt.show()


**3.1 First Floor Living Area**

The correlation coefficient between Sales Price and First-Floor Living Area is 0.62. This correlation suggests a moderate positive relationship between these variables. A higher First-Floor Living Area tends to be associated with a higher Sales Price, indicating a notable influence on property values.

The bar chart representing the mean Sales Price across different bins of First-Floor Living Area depicts an initial rise in the mean Sales Price with an increase in the living area. The graph demonstrates a steady ascent in the mean Sales Price until it reaches its peak within the 2500-3000 square feet range. However, beyond this point, there's a noticeable decline in the mean Sales Price.

The box plot analysis unveils an interesting trend in the distribution of Sales Price concerning First-Floor Living Area. Specifically, there's a considerable presence of outliers until the 2500 square feet mark. These outliers suggest variations in property values for first-floor living spaces up to this size.

![download](https://github.com/julietansy/Ames-Housing/assets/151416878/fe35e7be-5013-4ce7-a966-6ed3ac14953e)

        # Define the bin edges for grouping 1st_Flr_SF
        bin_edges = [data_final['1st_Flr_SF'].min()-1, 1000, 1500, 2000, 2500, 3000, 3500, float('inf')]
        bin_labels = ['Lowest', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000-3500', '3500+']
        
        # Create a new column "1st_Flr_SF_Group" with the corresponding area bin labels
        data_final['1st_Flr_SF_Group'] = pd.cut(data_final['1st_Flr_SF'], bins=bin_edges, labels=bin_labels)
        
        # Calculate the mean sales price by 1st_Flr_SF group
        mean_sales_price_by_1stflrsf_group = data_final.groupby("1st_Flr_SF_Group")["SalePrice"].mean().reset_index().round(2)
        
        # Sort the data by 1st_Flr_SF in ascending order
        mean_sales_price_by_1stflrsf_group = mean_sales_price_by_1stflrsf_group.sort_values("1st_Flr_SF_Group")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the mean sales price by 1st_Flr_SF group
        ax1.bar(mean_sales_price_by_1stflrsf_group["1st_Flr_SF_Group"], mean_sales_price_by_1stflrsf_group["SalePrice"])
        ax1.set_xlabel("1st_Flr_SF Group")
        ax1.set_ylabel("Mean Sales Price")
        ax1.set_title("Mean Sales Price by 1st_Flr_SF Group")
        
        # Create a box plot of Sales Price by 1st_Flr_SF Group
        sns.boxplot(x=data_final["1st_Flr_SF_Group"], y=data_final["SalePrice"], ax=ax2)
        ax2.set_xlabel("1st_Flr_SF Group")
        ax2.set_ylabel("Sales Price")
        ax2.set_title("Sales Price Distribution by 1st_Flr_SF Group")
        
        # Adjust the spacing between subplots
        fig.tight_layout()
        
        # Calculate the correlation between sales price and 1st_Flr_SF
        correlation = data_final['SalePrice'].corr(data_final['1st_Flr_SF'])
        
        # Print the correlation
        print("Correlation between Sales Price and 1st_Flr_SF: {:.2f}".format(correlation))
        
        # Show the plots
        plt.show()


**3.1 Ground Living Area**

The correlation coefficient between Sales Price and Ground Living Area stands at 0.71. This correlation implies a strong positive relationship between these variables. As the Ground Living Area increases, there's a notable tendency for the Sales Price to increase, indicating a substantial impact on property values.

The bar chart, portraying the mean Sales Price across different bins of Ground Living Area, exhibits a consistent rise in the mean Sales Price corresponding to an increase in the living area. Notably, the mean Sales Price steadily climbs and reaches its pinnacle within the 3000+ square feet range, suggesting that larger Ground Living Areas command higher property values.

Analyzing the box plot, it's evident that the upper whisker—representing the upper range of the data—is notably longer for the 3000+ square feet category. This indicates a wider range of property values and potential outliers for homes with larger Ground Living Areas. Moreover, there's a considerable presence of outliers observed within the 1500-2000 and 2000-2500 square feet categories, showcasing variations in Sales Prices within these specific ranges.

![download](https://github.com/julietansy/Ames-Housing/assets/151416878/337022ac-bcb6-424f-916b-1afc7bda890e)

        # Define the bin edges for grouping Gr_Liv_Area
        bin_edges = [data_final['Gr_Liv_Area'].min()-1, 1000, 1500, 2000, 2500, 3000, float('inf')]
        bin_labels = ['Lowest', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000+']
        
        # Create a new column "Gr_Liv_Area_Group" with the corresponding area bin labels
        data_final['Gr_Liv_Area_Group'] = pd.cut(data_final['Gr_Liv_Area'], bins=bin_edges, labels=bin_labels)
        
        # Calculate the mean sales price by Gr_Liv_Area group
        mean_sales_price_by_grlivarea_group = data_final.groupby("Gr_Liv_Area_Group")["SalePrice"].mean().reset_index().round(2)
        
        # Sort the data by Gr_Liv_Area in ascending order
        mean_sales_price_by_grlivarea_group = mean_sales_price_by_grlivarea_group.sort_values("Gr_Liv_Area_Group")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot the mean sales price by Gr_Liv_Area group
        ax1.bar(mean_sales_price_by_grlivarea_group["Gr_Liv_Area_Group"], mean_sales_price_by_grlivarea_group["SalePrice"])
        ax1.set_xlabel("Gr_Liv_Area Group")
        ax1.set_ylabel("Mean Sales Price")
        ax1.set_title("Mean Sales Price by Gr_Liv_Area Group")
        
        # Create a box plot of Sales Price by Gr_Liv_Area Group
        sns.boxplot(x=data_final["Gr_Liv_Area_Group"], y=data_final["SalePrice"], ax=ax2)
        ax2.set_xlabel("Gr_Liv_Area Group")
        ax2.set_ylabel("Sales Price")
        ax2.set_title("Sales Price Distribution by Gr_Liv_Area Group")
        
        # Adjust the spacing between subplots
        fig.tight_layout()
        
        # Calculate the correlation between sales price and Gr_Liv_Area
        correlation = data_final['SalePrice'].corr(data_final['Gr_Liv_Area'])
        
        # Print the correlation
        print("Correlation between Sales Price and Gr_Liv_Area: {:.2f}".format(correlation))
        
        # Show the plots
        plt.show()

**3.3 Total Basement Area**

The correlation coefficient between Sales Price and Basement Living Area stands at 0.63, indicating a moderately strong positive relationship between these variables. This relationship implies that as the Basement Living Area increases, there's a notable tendency for the Sales Price to increase as well.

The bar chart illustrates the mean Sales Price across different bins of Basement Living Area. It reveals a consistent increase in the mean Sales Price with an expansion in the basement area. The mean Sales Price reaches its peak within the 2500-3000 square feet range, suggesting that larger basement areas tend to command higher property values.

Analyzing the box plot, it's observed that there are more outliers within the range till 2000-2500 square feet. Interestingly, although the upper whiskers for the 2000-2500, 2500-3000, and 3000+ square feet bins appear similar, there are distinct variations in the mean Sales Price for these categories. Specifically, the lower whisker for the 2500-3000 square feet range presents a higher value compared to both the 2000-2500 and 3000+ ranges, despite their shared upper whisker lengths. Furthermore, the price range for the 2500-3000 square feet bin appears relatively narrower.

![download](https://github.com/julietansy/Ames-Housing/assets/151416878/d2f482dd-cec3-4155-b184-ab4e65d152c0)

        # Define the bin edges for grouping Total_Bsmt_SF
        bin_edges = [data_final['Total_Bsmt_SF'].min()-1, 1000, 1500, 2000, 2500, 3000, float('inf')]
        bin_labels = ['Lowest', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000+']
        
        # Create a new column "Total_Bsmt_SF_Group" with the corresponding area bin labels
        data_final['Total_Bsmt_SF_Group'] = pd.cut(data_final['Total_Bsmt_SF'], bins=bin_edges, labels=bin_labels)
        
        # Calculate the mean sales price by Total_Bsmt_SF group
        mean_sales_price_by_bsmtsf_group = data_final.groupby("Total_Bsmt_SF_Group")["SalePrice"].mean().reset_index().round(2)
        
        # Sort the data by Total_Bsmt_SF in ascending order
        mean_sales_price_by_bsmtsf_group = mean_sales_price_by_bsmtsf_group.sort_values("Total_Bsmt_SF_Group")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the mean sales price by Total_Bsmt_SF group
        ax1.bar(mean_sales_price_by_bsmtsf_group["Total_Bsmt_SF_Group"], mean_sales_price_by_bsmtsf_group["SalePrice"])
        ax1.set_xlabel("Total_Bsmt_SF Group")
        ax1.set_ylabel("Mean Sales Price")
        ax1.set_title("Mean Sales Price by Total_Bsmt_SF Group")
        
        # Create a box plot of Sales Price by Total_Bsmt_SF Group
        sns.boxplot(x=data_final["Total_Bsmt_SF_Group"], y=data_final["SalePrice"], ax=ax2)
        ax2.set_xlabel("Total_Bsmt_SF Group")
        ax2.set_ylabel("Sales Price")
        ax2.set_title("Sales Price Distribution by Total_Bsmt_SF Group")
        
        # Adjust the spacing between subplots
        fig.tight_layout()
        
        # Calculate the correlation between sales price and Total_Bsmt_SF
        correlation = data_final['SalePrice'].corr(data_final['Total_Bsmt_SF'])
        
        # Print the correlation
        print("Correlation between Sales Price and Total_Bsmt_SF: {:.2f}".format(correlation))
        
        # Show the plots
        plt.show()

**4 Garage Area**

The correlation between Sales Price and Garage Area stands at 0.63, indicating a moderately strong positive relationship between these variables. This correlation suggests that as the Garage Area increases, there's a noticeable tendency for the Sales Price to rise as well, pointing to the influence of garage size on property values.

When examining the bar chart representing the mean Sales Price across different Garage Area bins, a consistent increase in the mean Sales Price is observed with the expansion of the garage area. Notably, the mean Sales Price reaches its peak within the 1000-1200 square feet range, indicating that properties with garage areas falling within this range tend to command higher values.

Analyzing the box plot reveals insightful outliers present until the 1000 square feet mark. However, there's a scarcity of data points for garage areas exceeding 1400 square feet, with the range showing minimal or no representation. This scarcity might suggest a less common occurrence of properties with extremely large garage spaces within the dataset.

![download](https://github.com/julietansy/Ames-Housing/assets/151416878/2ae784fa-d808-48af-995b-f3852398ced8)

        # Define the bin edges for grouping Garage_Area
        data_final['Garage_Area'] = pd.to_numeric(data_final['Garage_Area'], errors='coerce')
        bin_edges = [data_final['Garage_Area'].min()-1, 200, 400, 600, 800, 1000, 1200, 1400, float('inf')]
        bin_labels = ['Lowest', '200-400', '400-600', '600-800', '800-1000', '1000-1200', '1200-1400', '1400+']
        
        # Create a new column "Garage_Area_Group" with the corresponding area bin labels
        data_final['Garage_Area_Group'] = pd.cut(data_final['Garage_Area'], bins=bin_edges, labels=bin_labels)
        
        # Calculate the mean sales price by Garage_Area group
        mean_sales_price_by_garagearea_group = data_final.groupby("Garage_Area_Group")["SalePrice"].mean().reset_index().round(2)
        
        # Sort the data by Garage_Area in ascending order
        mean_sales_price_by_garagearea_group = mean_sales_price_by_garagearea_group.sort_values("Garage_Area_Group")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the mean sales price by Garage_Area group
        ax1.bar(mean_sales_price_by_garagearea_group["Garage_Area_Group"], mean_sales_price_by_garagearea_group["SalePrice"])
        ax1.set_xlabel("Garage Area Group")
        ax1.set_ylabel("Mean Sales Price")
        ax1.set_title("Mean Sales Price by Garage Area Group")
        
        # Create a box plot of Sales Price by Garage Area Group
        sns.boxplot(x=data_final["Garage_Area_Group"], y=data_final["SalePrice"], ax=ax2)
        ax2.set_xlabel("Garage Area Group")
        ax2.set_ylabel("Sales Price")
        ax2.set_title("Sales Price Distribution by Garage Area Group")
        
        # Adjust the spacing between subplots
        fig.tight_layout()
        
        # Calculate the correlation between sales price and Garage_Area
        correlation = data_final['SalePrice'].corr(data_final['Garage_Area'])
        
        # Print the correlation
        print("Correlation between Sales Price and Garage Area: {:.2f}".format(correlation))
        
        # Show the plots
        plt.show()


**5. Full Bath**

The correlation between Sales Price and the Total Number of Full Baths stands at 0.59, indicating a moderately positive relationship between these variables. This correlation suggests that as the number of total full baths increases, there's a noticeable tendency for the Sales Price to rise, underscoring the influence of bathroom count on property values.

Examining the bar chart representing the mean Sales Price across different counts of total full baths, a consistent ascent in the mean Sales Price is observed with an increase in the number of baths. Notably, the mean Sales Price reaches its peak at the 4-bath mark, indicating that properties with four full baths tend to command higher values.

The box plot analysis highlights intriguing outliers within the 1 to 3 full bath range. Moreover, the upper whisker for properties with 4 baths demonstrates notably higher values compared to other bath counts, suggesting a distinct premium associated with homes having this specific bathroom count.

![download](https://github.com/julietansy/Ames-Housing/assets/151416878/c2bd8cb5-e42c-4564-983f-edc9857517a7)

        # Calculate the total number of full baths (Full_Bath + Bsmt_Full_Bath)
        data_final['Full_Bath'] = pd.to_numeric(data_final['Full_Bath'], errors='coerce')
        data_final['Bsmt_Full_Bath'] = pd.to_numeric(data_final['Bsmt_Full_Bath'], errors='coerce')
        data_final['Total_Full_Bath'] = data_final['Full_Bath'] + data_final['Bsmt_Full_Bath']
        
        # Define the bin edges for grouping Total_Full_Bath
        bin_edges = [data_final['Total_Full_Bath'].min()-1, 1, 2, 3, 4, float('inf')]
        bin_labels = ['1', '2', '3', '4', '5+']
        
        # Create a new column "Total_Full_Bath_Group" with the corresponding bin labels
        data_final['Total_Full_Bath_Group'] = pd.cut(data_final['Total_Full_Bath'], bins=bin_edges, labels=bin_labels)
        
        # Calculate the mean sales price by Total_Full_Bath group
        mean_sales_price_by_total_fullbath_group = data_final.groupby("Total_Full_Bath_Group")["SalePrice"].mean().reset_index().round(2)
        
        # Sort the data by Total_Full_Bath in ascending order
        mean_sales_price_by_total_fullbath_group = mean_sales_price_by_total_fullbath_group.sort_values("Total_Full_Bath_Group")

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the mean sales price by Total_Full_Bath group
        ax1.bar(mean_sales_price_by_total_fullbath_group["Total_Full_Bath_Group"], mean_sales_price_by_total_fullbath_group["SalePrice"])
        ax1.set_xlabel("Total Full Bath Group")
        ax1.set_ylabel("Mean Sales Price")
        ax1.set_title("Mean Sales Price by Total Full Bath Group")
        
        # Create a box plot of Sales Price by Total Full Bath Group
        sns.boxplot(x=data_final["Total_Full_Bath_Group"], y=data_final["SalePrice"], ax=ax2)
        ax2.set_xlabel("Total Full Bath Group")
        ax2.set_ylabel("Sales Price")
        ax2.set_title("Sales Price Distribution by Total Full Bath Group")
        
        # Adjust the spacing between subplots
        fig.tight_layout()
        
        # Calculate the correlation between sales price and Total_Full_Bath
        correlation = data_final['SalePrice'].corr(data_final['Total_Full_Bath'])
        
        # Print the correlation
        print("Correlation between Sales Price and Total Full Bath: {:.2f}".format(correlation))
        
        # Show the plots
        plt.show()



## Conclusion
**Overall Quality:** The quality of a house strongly influences its sales price. Properties with higher quality ratings tend to command higher prices. However, exceptional houses within the same quality category might have significantly higher values due to other influential features.

**Sold House Age:** Younger houses generally have higher sale prices compared to older ones. While the age of the house plays a role, other influential factors contribute to the variation in prices across different age groups.

**Living Area:**

**Total Living Area: **Larger living spaces positively impact property values, with a consistent rise in prices up to a certain threshold, beyond which there's a decline in the mean sales price.
First-Floor Living Area, Ground Living Area, and Total Basement Area: Similar to Total Living Area, larger areas in these specific sections positively influence property values, with larger areas generally commanding higher prices. However, variations exist within certain ranges, suggesting nuanced effects on sales prices based on specific area sizes.
Garage Area: The size of the garage area also correlates positively with sales price. Larger garage areas tend to be associated with higher property values. However, there's a scarcity of data for extremely large garage spaces, indicating their relative rarity in the dataset.

**Full Bath Count:** Properties with more full baths generally command higher prices. However, the premium associated with 4 baths stands out, suggesting a distinct increase in value for homes with this specific bathroom count.


## Recommendations


**Further Investigation:** Explore potential interactions between these features to understand how they collectively influence property values.

**Account for Rarity:** Properties with extremely large living areas or garage spaces might need special consideration due to their rarity and potential impact on pricing.

**Outlier Analysis: **Further investigate outliers within each feature category to understand the unique characteristics of properties commanding exceptionally higher prices.

By leveraging these insights, real estate stakeholders can better understand the factors influencing property values and make informed decisions regarding pricing, investment, and market strategies.






________________________________________________________________________________________________________________
## Appendix

## Data Cleaning

### Removal of Irrelevant Data

The 'Order' and 'PID' columns were removed from the dataset.

        # Drop "Order" and "PID" variables from the dataset
        data = data.drop(["Order", "PID"], axis=1)


No duplicated transactions were found.

        # Check for duplicated rows 
        if data.duplicated().sum() == 0:
            print("No change in dataset as there are no duplicated row.")
        else:
            # Keep row with latest built year
            data = data.sort_values('YearBuilt').drop_duplicates(keep='last')
            print("Number of duplicated rows removed: ", data.duplicated().sum())
            print("After removing duplicated rows, dataset contains ", data.shape[0], " rows and ", data.shape[1], " columns")

### Correction of Incorrect Data or Null Values
    
An incorrect input for Garage_Yr_Blt was identified as '2207' and was replaced with null.

        # Remove incorrect input of data, tentatively remove value.
        data['Garage_Yr_Blt'] = data['Garage_Yr_Blt'].replace([2207], np.nan)
        
        # Verify abnormal garage year build is being removed
        np.sort(data["Garage_Yr_Blt"].unique())

Identified null values within the dataset show a high percentage in various variables like 'Pool', 'Alley', 'Fence', 'Fireplace', etc. These likely signify the absence of such features in the properties. The variables shall be investigated individually to replace null value in the subsequent data handling.

        # Find null values in each variable
        null_values = data.isnull().sum()
        
        # Calculate the percentage of null values rounded to 2 decimal places
        null_percentages = (null_values / len(data) * 100).round(2)
        
        # Find the data types for variables with null values
        null_data_types = data.dtypes[null_values > 0]
        
        # Combine the counts and percentages into a single DataFrame
        null_summary = pd.concat([null_values, null_percentages, null_data_types], axis=1)
        null_summary.columns = ['Null Count', '% Null', 'Data Type']
        
        # Sort the DataFrame by the percentage of null values in descending order
        null_summary = null_summary[null_summary['Null Count'] > 0].sort_values('% Null', ascending=False)
        
        # Print the combined summary
        print(null_summary.to_string())
        
        # The high percentage of null most likely represent an absence of a feature


    ---- Output ----
                    Null Count  % Null Data Type
      
    Pool_QC               2917   99.56    object
    Misc_Feature          2824   96.38    object
    Alley                 2732   93.24    object
    Fence                 2358   80.48    object
    Fireplace_Qu          1422   48.53    object
    Lot_Frontage           490   16.72   float64
    Garage_Yr_Blt          160    5.46   float64
    Garage_Cond            159    5.43    object
    Garage_Qual            159    5.43    object
    Garage_Finish          159    5.43    object
    Garage_Type            157    5.36    object
    Bsmt_Exposure           83    2.83    object
    BsmtFin_Type_2          81    2.76    object
    BsmtFin_Type_1          80    2.73    object
    Bsmt_Qual               80    2.73    object
    Bsmt_Cond               80    2.73    object
    Mas_Vnr_Area            23    0.78   float64
    Mas_Vnr_Type            23    0.78    object
    Bsmt_Half_Bath           2    0.07   float64
    Bsmt_Full_Bath           2    0.07   float64
    Total_Bsmt_SF            1    0.03   float64
    Bsmt_Unf_SF              1    0.03   float64
    Garage_Cars              1    0.03   float64
    Garage_Area              1    0.03   float64
    BsmtFin_SF_2             1    0.03   float64
    BsmtFin_SF_1             1    0.03   float64
    Electrical               1    0.03    object
    ----- ----- -----

**Handling Null Values for Pool**
        
        # Check for missing values in Pool_QC
        pool_qc_null = data[data["Pool_QC"].isnull()] 
        pool_area = pool_qc_null['Pool_Area'] 
        print('When Pool QC is null, the value of Pool Area is ', np.unique(pool_area), '\nIndicating an absence of pool. Hence, it shall be replace with "None".')
        
        # Replace all 'Pool_QC' with 'None' when 'Pool_Area' is null
        data['Pool_QC'] = data['Pool_QC'].fillna("None")

    ---- Output ----
    When Pool QC is null, the value of Pool Area is  [0] 
    Indicating an absence of pool. Hence, it shall be replace with "None".  
    ----- ----- -----

**Handling Null Values for Misc Feature**

        # Check for missing values in Misc_Feature
        misc = data[data["Misc_Feature"].isnull()]
        misc_value = misc['Misc_Val']
        print('When there is no Misc Feature, the misc value (Misc Val) is', np.unique(misc_value), '\nIndicating an absence of MiscFeature. Hence, it shall be replace with "None".')
        
        # Replace all 'Misc_Feature' with 'None' when 'Misc_Val' is null
        data['Misc_Feature'] = data['Misc_Feature'].fillna("None")

    ---- Output ----
    When there is no Misc Feature, the misc value (Misc Val) is [0] 
    Indicating an absence of MiscFeature. Hence, it shall be replace with "None".
    ----- ----- -----

**Handling Null Values for Alley and Fence**

        # Since Alley and Fence does not have other related variables, null in said field reflects absence of feature
        # Replace all missing null values with 'None' to represent absence
        data['Alley'] = data['Alley'].fillna("None")
        data['Fence'] = data['Fence'].fillna("None")


**Handling Null Values for Fireplace**


        # Check for missing values in Fireplace_Qu
        fireplace_quality = data[data["Fireplace_Qu"].isnull()]
        fireplace_no = fireplace_quality['Fireplaces']
        print('When Fireplace Quality is null,the number of fireplace is', np.unique(fireplace_no), ',\nindicating an absence of fireplace. Hence, Fireplace Qu shall be replace with None.')
        
        # Replace all 'Fireplace_Qu' with 'None' when 'Fireplace' is absence
        data['Fireplace_Qu'] = data['Fireplace_Qu'].fillna("None")
    
    ---- Output ----
    When Fireplace Quality is null,the number of fireplace is [0] ,
    indicating an absence of fireplace. Hence, Fireplace Qu shall be replace with None.
    ----- ----- -----



**Handling Null Values for Lot Frontage**

Size of lot frontage are influenced by multiple factors, these factors includes the neighbourhood the property is located in, the lot shape and the lot configuration. Hence, in order to get a more representative value, these factors were grouped together in order to determine the mean value used to replace the null value.

        # Find non-null values of 'Lot_Frontage'
        non_null_lot_frontage = data[data['Lot_Frontage'].notnull()]
        
        # Calculate the global median of 'Lot_Frontage'
        global_mean = data['Lot_Frontage'].mean()
        
        # Group by ['Neighborhood', 'Lot_Shape', 'Lot_Config'] and calculate mean of 'Lot_Frontage'
        grouped_data = non_null_lot_frontage.groupby(['Neighborhood', 'Lot_Shape', 'Lot_Config'])
        mean_lot_frontage_grouped = grouped_data['Lot_Frontage'].mean()
        
        # Replace null values of 'Lot_Frontage' using group means
        data['Lot_Frontage'] = data.apply(lambda row: mean_lot_frontage_grouped.get((row['Neighborhood'], row['Lot_Shape'], row['Lot_Config']), global_mean)
                                          if pd.isnull(row['Lot_Frontage'])
                                          else row['Lot_Frontage'], axis=1).round(0)
        
        # Replace null values of 'Lot_Frontage' with mean of entire dataset if any group has all null values
        if data['Lot_Frontage'].isnull().any():
            data['Lot_Frontage'].fillna(data['Lot_Frontage'].mean().round(0), inplace=True)


**Handling Null Values for Garage Type**

In scenarios where 'Garage_Type' is null, it typically signifies the absence of a garage on the property. Consequently, other garage-related variables should align with this absence and reflect null values. In the code provided, most variables—except for 'Garage_Cars' and 'Garage_Area'—show a complete absence of non-null values when 'Garage_Type' is null.

However, for 'Garage_Cars' and 'Garage_Area', while they indicate non-null counts, the aggregation statistics show values as 0.0. This discrepancy arises because even though these variables contain non-null entries, those entries are deliberately set to 0.0 to signify the lack of a garage. This approach ensures consistency throughout the dataset, clearly denoting that when 'Garage_Type' is null (indicating no garage), all related variables are adjusted to explicitly represent the absence of a garage, rather than retaining potentially misleading default or random values.


        # Group garage variables
        garage_cols = ['Garage_Yr_Blt', 'Garage_Type', 'Garage_Finish', 'Garage_Cars', 
                       'Garage_Area', 'Garage_Qual', 'Garage_Cond']
        
        # Select rows where all values in garage_cols are either 0 or null, representing absence of garage
        selected_rows = data[(data[garage_cols].eq(0) | data[garage_cols].isnull()).all(axis=1)]
        
        # Replace selected rows in the original dataset with 0 or 'None' for garage-related columns
        data.loc[selected_rows.index, garage_cols] = data.loc[selected_rows.index, garage_cols].fillna(0).replace(0, 'None')
        
        # Locate the row with null values in garage-related columns
        row_with_null_garage = data[data[garage_cols].isnull().any(axis=1)]
        print("Row with null values in garage-related columns:")
        print(row_with_null_garage[garage_cols])
        
        # Replace the value in row 1356's and 21236's garage_cols with 'None' or 0
        # Since all Garage related information are not available, it is likely an error input
        for col in garage_cols:
            if data[col].dtype == 'object':
                data.at[1356, col] = 'None'
                data.at[2236, col] = 'None'
            else:
                data.at[1356, col] = 0
                data.at[2236, col] = 0

    ---- Output ----
    Row with null values in garage-related columns:
         Garage_Yr_Blt Garage_Type Garage_Finish Garage_Cars Garage_Area  \
    1356           NaN      Detchd           NaN         1.0       360.0   
    2236           NaN      Detchd           NaN         NaN         NaN   
    2260           NaN      Attchd           RFn         2.0       502.0   
    
         Garage_Qual Garage_Cond  
    1356         NaN         NaN  
    2236         NaN         NaN  
    2260          TA          TA  
    ----- ----- -----
    

        # Row 2260 had missing Garage_Yr_Blt
        # Count occurrences where Garage_Yr_Blt is equal to Year_Built
        match_year = data[data['Garage_Yr_Blt'] == data['Year_Built']].shape[0]
        match_percentage = (match_year / data.shape[0]) * 100
        print("\nPercentage of occurrences where Garage_Yr_Blt is equal to Year_Built: {:.2f}%".format(match_percentage))
        
        # To replace missing values in Garage_Yr_Blt with Year_Built
        data['Garage_Yr_Blt'] = data['Garage_Yr_Blt'].fillna(data['Year_Built'])

    ---- Output ----
    Percentage of occurrences where Garage_Yr_Blt is equal to Year_Built: 76.01%
    ----- ----- -----

**Handling Null Values related to Mas_Vnr_Area and Mas_Vnr_Type**

Based on the inital null count, both variables had the same count of 23. As per the output the code, Mas_Vnr_Type had 23 null values when Mas_Vnr_Area is null. Hence, all occurence of nulls in Mas_Vnr_Area are nulls for Mas_Vnr_Type too. With that, the null for Mas_Vnr_Area and Mas Vnr_Type shall be replaced with 'None' and '0' respectively.

        # Print Mas_Vnr_Type when Mas_Vnr_Area is null
        null_mas_vnr_type = data[data['Mas_Vnr_Area'].isnull()]['Mas_Vnr_Type'].isnull()
        print(null_mas_vnr_type.count(), ' Mas_Vnr_Type are null')
        
        # Replace Mas_Vnr_Type as 'None' when Mas_Vnr_Area is null
        data['Mas_Vnr_Type'] = data['Mas_Vnr_Type'].fillna('None')
        
        # Replacing null for Mas_Vnr_Area as 0 as it represents absence 
        data['Mas_Vnr_Area'] = data['Mas_Vnr_Area'].fillna('0')

    ---- Output ----
    23  Mas_Vnr_Type are null
    ----- ----- -----

**Handling Null Values for Electrical**

As there is only 1 null value for Electical in the entire dataset, it shall be replaced with mode since it is a categorical variable.

        # Find mode of Electrical
        mode_value = data['Electrical'].mode()[0]
        
        # Replace all null 'Electrical' with mode
        data['Electrical'] = data['Electrical'].fillna(mode_value)
        
        # Verify that all null value had been replaced
        print("\033[1;96mElectical Type after replacing null value\033[0m")
        print(data['Electrical'].unique())

    ---- Output ----
    Electical Type after replacing null value
    ['SBrkr' 'FuseA' 'FuseF' 'FuseP' 'Mix']
    ----- ----- -----

**Handling Null Values for Basement**

        # Group basement (bsmt) variables
        bsmt_cols = ['Total_Bsmt_SF', 'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure', 
                     'BsmtFin_Type_1', 'BsmtFin_SF_1', 'BsmtFin_Type_2', 'BsmtFin_SF_2', 
                     'Bsmt_Unf_SF', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath']
        
        # Select rows where all values in bsmt_cols are either 0 or null, representing absence of basement
        selected_rows = data[(data[bsmt_cols].eq(0) | data[bsmt_cols].isnull()).all(axis=1)]
        
        # Replace selected rows in the original dataset with 0 or 'None' for basement-related columns
        data.loc[selected_rows.index, bsmt_cols] = data.loc[selected_rows.index, bsmt_cols].fillna(0).replace(0, 'None')
        
        # Locate the row with null values in basement-related columns
        row_with_null_basement = data[data[bsmt_cols].isnull().any(axis=1)]
        print("Row with null values in basement-related columns:")
        print(row_with_null_basement[bsmt_cols])

    ---- Output ----
    Row with null values in basement-related columns:
         Total_Bsmt_SF Bsmt_Qual Bsmt_Cond Bsmt_Exposure BsmtFin_Type_1  \
    66          1595.0        Gd        TA           NaN            Unf   
    444         3206.0        Gd        TA            No            GLQ   
    1796         725.0        Gd        TA           NaN            Unf   
    2779         936.0        Gd        TA           NaN            Unf   
    
         BsmtFin_SF_1 BsmtFin_Type_2 BsmtFin_SF_2 Bsmt_Unf_SF Bsmt_Full_Bath  \
    66            0.0            Unf          0.0      1595.0            0.0   
    444        1124.0            NaN        479.0      1603.0            1.0   
    1796          0.0            Unf          0.0       725.0            0.0   
    2779          0.0            Unf          0.0       936.0            0.0   
    
         Bsmt_Half_Bath  
    66              0.0  
    444             0.0  
    1796            0.0  
    2779            0.0  
    ----- ----- -----

        # Find mode of Bsmt_Exposure
        mode_value = data['Bsmt_Exposure'].mode()[0]
        
        # Replace all null 'Bsmt_Exposure' with mode
        data['Bsmt_Exposure'] = data['Bsmt_Exposure'].fillna(mode_value)
        
        # Verify that all null value had been replaced
        print("\033[1;96mBasement exposure after replacing null value\033[0m")
        print(data['Bsmt_Exposure'].unique())
        
    ---- Output ----
    Basement exposure after replacing null value
    ['Gd' 'No' 'Mn' 'Av' 'None']
   ----- ----- -----

        # Locate the row with null values in basement-related columns
        row_with_null_basement = data[data[bsmt_cols].isnull().any(axis=1)]
        print("Row with null values in basement-related columns:")
        print(row_with_null_basement[bsmt_cols])

    ---- Output ----
    Row with null values in basement-related columns:
        Total_Bsmt_SF Bsmt_Qual Bsmt_Cond Bsmt_Exposure BsmtFin_Type_1  \
    444        3206.0        Gd        TA            No            GLQ   
    
        BsmtFin_SF_1 BsmtFin_Type_2 BsmtFin_SF_2 Bsmt_Unf_SF Bsmt_Full_Bath  \
    444       1124.0            NaN        479.0      1603.0            1.0   
    
        Bsmt_Half_Bath  
    444            0.0 
   ----- ----- -----

            # Filter and group the data based on neighborhood and year_built
        neighborhood = data.loc[444, 'Neighborhood']
        year_built = data.loc[444, 'Year_Built']
        group_data = data[(data['Neighborhood'] == neighborhood) & (data['Year_Built'] == year_built)]
        
        # Calculate mode of missing variables, groupbed by neighborhood and garage year build
        mode_bsmtfin_type_2 = group_data['BsmtFin_Type_2'].mode()
        
        # Assign the mode value to a specific row (e.g., 444) for BsmtFin_Type_2
        data.loc[444, 'BsmtFin_Type_2'] = mode_bsmtfin_type_2[0]
        
        # Print the updated values for row 444
        print("\n\n\033[1;96mRow 444 after updating\033[0m")
        print(data.loc[444, bsmt_cols])

    ---- Output ----
    Row 444 after updating
    Total_Bsmt_SF     3206.0
    Bsmt_Qual             Gd
    Bsmt_Cond             TA
    Bsmt_Exposure         No
    BsmtFin_Type_1       GLQ
    BsmtFin_SF_1      1124.0
    BsmtFin_Type_2       Unf
    BsmtFin_SF_2       479.0
    Bsmt_Unf_SF       1603.0
    Bsmt_Full_Bath       1.0
    Bsmt_Half_Bath       0.0
    Name: 444, dtype: object
    
