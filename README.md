<h1> SINGAPORE RESALE FLAT PREDICTION
	

![image](https://github.com/user-attachments/assets/197858b9-bded-479b-ad66-9ba28ec8af25)


The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration. This project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

The project will involve the following tasks:

						1.Data Collection and Preprocessing: Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.
	  
						2.Feature Engineering: Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.
	  
						3.Model Selection and Training: Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.
	  
						4.Model Evaluation: Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.
	  
						5.Streamlit Web Application: Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.
	  
						6.Deployment on Render: Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.
	  
						7.Testing and Validation: Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.


GLIMPSE OF THE PROJECT:

Import all the files using read_csv 

	df1=pd.read_csv("C:/Users/yaazhisai/Desktop/singapore resale/ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")
	df2=pd.read_csv("C:/Users/yaazhisai/Desktop/singapore resale/ResaleFlatPricesBasedonApprovalDate19901999.csv")
	df3=pd.read_csv("C:/Users/yaazhisai/Desktop/singapore resale/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
	df4=pd.read_csv("C:/Users/yaazhisai/Desktop/singapore resale/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")
	df5=pd.read_csv("C:/Users/yaazhisai/Desktop/singapore resale/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")
 
Concat all the files in to single dataframe.

	df_new1=pd.concat([df2,df1,df5,df3,df4],axis=0,ignore_index=True)
 
view the first 2 rows of df.

	df_new1.head(2)
	month	town	flat_type	block	street_name	storey_range	floor_area_sqm	flat_model	lease_commence_date	resale_price	remaining_lease
	0	1990-01	ANG MO KIO	1 ROOM	309	ANG MO KIO AVE 1	10 TO 12	31.0	IMPROVED	1977	9000.0	NaN
	1	1990-01	ANG MO KIO	1 ROOM	309	ANG MO KIO AVE 1	04 TO 06	31.0	IMPROVED	1977	6000.0	NaN

Do the necessary Data cleaning(EDA) and preprocessing steps.

check for skewness and outliers 

convert categorical values to numerical values using LabelEncoder

once cleeaning is done,split x(features) and y(target) from dataframe

split the training and testing data

choose the appropriate model and train the model.

If hyperparameters are there,find the best one to get maximum Accuracy.

once training is done,check for MSE,MAE,RMSE and ACCURACY OF  ALL THE MODEL.

Save the model

In this project,I have trained these models

	{'MODEL': <class 'sklearn.tree._classes.DecisionTreeRegressor'>, 'MEAN_ABSOLUTE_ERROR': 0.06890150346699297, 'MEAN_SQUARE_ERROR': 0.010058328573748481, 'ROOT_SQUARE_ERROR': 0.10029121882671724, 'ACCURACY': (99.78327952089737, '%')}
	{'MODEL': <class 'sklearn.ensemble._forest.ExtraTreesRegressor'>, 'MEAN_ABSOLUTE_ERROR': 0.05984596604531716, 'MEAN_SQUARE_ERROR': 0.007321654340707591, 'ROOT_SQUARE_ERROR': 0.08556666606049106, 'ACCURACY': (99.78327952089737, '%')}
	{'MODEL': <class 'sklearn.ensemble._forest.RandomForestRegressor'>, 'MEAN_ABSOLUTE_ERROR': 0.05523103983816146, 'MEAN_SQUARE_ERROR': 0.006211495832529426, 'ROOT_SQUARE_ERROR': 0.07881304354311808, 'ACCURACY': (99.58860309031743, '%')}
	{'MODEL': <class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>, 'MEAN_ABSOLUTE_ERROR': 0.1813596153357633, 'MEAN_SQUARE_ERROR': 0.051445079625975214, 'ROOT_SQUARE_ERROR': 0.22681507803930323, 'ACCURACY': (84.51657538748385, '%')}
	{'MODEL': <class 'sklearn.ensemble._gb.GradientBoostingRegressor'>, 'MEAN_ABSOLUTE_ERROR': 0.10224060793954298, 'MEAN_SQUARE_ERROR': 0.017974763503277154, 'ROOT_SQUARE_ERROR': 0.13406999479106857, 'ACCURACY': (94.56324408931292, '%')}
	{'MODEL': <class 'sklearn.neighbors._regression.KNeighborsRegressor'>, 'MEAN_ABSOLUTE_ERROR': 0.17079869658992694, 'MEAN_SQUARE_ERROR': 0.06315175423691703, 'ROOT_SQUARE_ERROR': 0.2513001278091936, 'ACCURACY': (88.45493982435923, '%')}


A user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.) and used DECISIONTREE MODEL TO 
trained machine learning model to predict the resale price based on user inputs.

Finally,deploy the Streamlit application on the Render platform to make it accessible to users over the internet.


Machine learning enables more accurate and data-driven predictions of resale prices across various industries, helping businesses and consumers make informed decisions.
Resale flat price prediction leverages machine learning to provide valuable insights, benefiting buyers, sellers, investors, and real estate professionals alike.











