\# House Price Prediction using Machine Learning



A Machine Learning project that predicts house prices based on property features using regression algorithms and provides an interactive web interface built with Streamlit.



\## Project Overview



House price prediction is a common Machine Learning problem where various property characteristics are used to estimate the expected price of a house.



This project performs:



\- Data loading and preprocessing

\- Data cleaning

\- Exploratory Data Analysis (EDA)

\- Feature engineering

\- Categorical variable encoding

\- Machine Learning model training

\- Model evaluation

\- Feature importance analysis

\- House price prediction

\- Streamlit web application deployment



The final application allows users to enter house details and receive an estimated house price.



\## Objectives



The main objectives of this project are:



1\. Analyze housing data and identify important factors affecting house prices.

2\. Clean and preprocess the dataset for Machine Learning.

3\. Perform Exploratory Data Analysis to understand relationships between features.

4\. Engineer useful features from the available data.

5\. Train regression models for house price prediction.

6\. Evaluate and compare Machine Learning models.

7\. Build an interactive Streamlit application for real-time predictions.



\## Dataset



The project uses a housing dataset containing information about residential properties.



\### Features



| Feature | Description |

|---|---|

| `area` | Area of the house in square feet |

| `bedrooms` | Number of bedrooms |

| `bathrooms` | Number of bathrooms |

| `stories` | Number of stories |

| `mainroad` | Whether the property has main road access |

| `guestroom` | Whether the property has a guest room |

| `basement` | Whether the property has a basement |

| `hotwaterheating` | Whether hot water heating is available |

| `airconditioning` | Whether air conditioning is available |

| `parking` | Number of parking spaces |

| `prefarea` | Whether the property is located in a preferred area |

| `furnishingstatus` | Furnishing status of the property |

| `price` | Target variable representing house price |



\## Exploratory Data Analysis



Several visualizations were performed to understand the dataset and relationships between variables.



\### Analysis included:



\- Distribution of house prices

\- Area vs. house price

\- Bedrooms vs. house price

\- Correlation analysis

\- Correlation heatmap

\- Feature relationships

\- Feature importance



\## Data Preprocessing



The following preprocessing techniques were applied:



\- Handling missing values

\- Removing duplicate records

\- Encoding categorical variables

\- Converting binary categorical variables into numerical values

\- Splitting the dataset into training and testing sets



The dataset was divided into:



\- \*\*80% Training Data\*\*

\- \*\*20% Testing Data\*\*



\## Feature Engineering



A derived feature was created:



\### Bedrooms per Area



```text

Bedrooms\_per\_area = bedrooms / area

This feature represents the relationship between the number of bedrooms and the size of the property.



Price\_per\_area was intentionally excluded from the final predictive model because it directly depends on the target variable price and would introduce target leakage.



&#x20;**Machine Learning Models**



The project experimented with multiple regression algorithms:



1\. Linear Regression

A basic regression algorithm used to model the relationship between input features and house prices.



2\. Ridge Regression

A regularized version of Linear Regression that helps reduce overfitting.



3\. Random Forest Regression

An ensemble Machine Learning algorithm that combines multiple decision trees to improve prediction performance.



The Random Forest model was selected for the final prediction application.



&#x20;**Model Evaluation**



The models were evaluated using:



Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted prices.



Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error.



R² Score

Measures how well the model explains the variation in house prices.



&#x20;**Streamlit Web Application**



The trained Machine Learning model is integrated into an interactive Streamlit web application.



Users can enter:



Area

Number of bedrooms

Number of bathrooms

Number of stories

Parking spaces

Main road access

Guest room availability

Basement availability

Hot water heating

Air conditioning

Preferred area

Furnishing status



The application then generates an estimated house price.



&#x20;**Running the Application**

1\. Clone the repository

git clone https://github.com/YOUR-USERNAME/House-Price-Prediction-ML.git

2\. Navigate into the project

cd House-Price-Prediction-ML

3\. Install dependencies

python -m pip install -r requirements.txt

4\. Run the Streamlit application

python -m streamlit run app.py



The application will open in your browser.



📁 Project Structure

House-Price-Prediction-ML/

│

├── data/

│   └── housing.csv

│

├── models/

│   ├── house\_price\_model.pkl

│   └── model\_columns.pkl

│

├── notebooks/

│   └── house\_price\_analysis.ipynb

│

├── app.py

├── requirements.txt

├── .gitignore

└── README.md



&#x20;**Technologies Used**

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Joblib

Jupyter Notebook

Streamlit

Git \& GitHub



&#x20;**Key Learning Outcomes**



Through this project, the following concepts were implemented:



Data preprocessing

Exploratory Data Analysis

Data visualization

Feature engineering

Categorical encoding

Regression

Ensemble Machine Learning

Model evaluation

Model serialization

Streamlit application development

Git and GitHub version control



&#x20;**Future Improvements**



Future versions of this project could include:



Hyperparameter tuning

Additional regression algorithms

Cross-validation

Advanced feature engineering

Location-based features

Interactive visual analytics

Cloud deployment

Improved UI/UX

Model monitoring



