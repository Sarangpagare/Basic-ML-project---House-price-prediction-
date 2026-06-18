from asyncio import graph
import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('train.csv')

# Clean numeric columns
for column in data.select_dtypes(include=['int64', 'float64']).columns:
    data[column] = data[column].fillna(data[column].mean())

# Clean text columns
for column in data.select_dtypes(include=['object','string']).columns:
    data[column] = data[column].fillna(data[column].mode()[0])        

x = data[[ 'OverallQual',
           'GrLivArea',
           'GarageCars',
           'GarageArea',
           'TotalBsmtSF',
           '1stFlrSF',
           'FullBath',
           'YearBuilt'   ]]

#  convert text to numeric using one-hot encoding
# X = pd.get_dummies(x, columns=['HouseStyle'], drop_first=True)

# our target variable
y = data['SalePrice']

# print(X.isnull().sum())

print(data.shape)



# plt.scatter(data['GrLivArea'], data['SalePrice'])
# plt.xlabel("Living Area")
# plt.ylabel("Sale Price")
# plt.title("Living Area vs Sale Price")
# plt.show()

# print(data.describe())
# print(
#     data.corr(numeric_only=True)['SalePrice']
#     .sort_values(ascending=False)
# )

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=42)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# model.fit(x_train , y_train)
# prediction = model.predict(x_test)
# error = mean_absolute_error(y_test, prediction)

param_grid = {
    'n_estimators':[50,100],
    'max_depth':[5 , 10 , 15],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
# Train all combinations
grid_search.fit(x_train, y_train)
# Best model
best_model = grid_search.best_estimator_

joblib.dump(best_model, 'house_price_model.pkl')

# predict and evaluate the best model1
prediction = best_model.predict(x_test)
error = mean_absolute_error(y_test, prediction)


scores = cross_val_score(
    model , x , y , cv=5 , scoring='neg_mean_absolute_error'    
)  

scores = -scores

print("Cross-Validation MAE Scores: ")
print(scores)

print("after average MAE: ")
print(scores.mean())

print("Best Parameters:")
print(grid_search.best_params_)

print("Best MAE:")
print(error)

# Feature Importance
# importance = model.feature_importances_

# feature_names = x.columns

# for feature, score in zip(feature_names, importance):
#     print(feature, ":", score)

# Visualize feature importance
# plt.figure(figsize=(10, 5))
# plt.bar(feature_names, importance)
# plt.xlabel("Features")
# plt.ylabel("Importance Score")
# plt.title("Feature Importance - Random Forest")
# plt.show()    


 # Take user input
OverallQual = float(input("Enter Overall Quality: "))
GrLivArea = float(input("Enter Living Area: "))
GarageCars = int(input("Enter Garage Cars: "))
GarageArea = float(input("Enter Garage Area: "))
TotalBsmtSF = float(input("Enter Total Basement SF: "))
FirstFlrSF = float(input("Enter 1st Floor SF: "))
FullBath = int(input("Enter Full Bathrooms: "))
YearBuilt = int(input("Enter Year Built: "))


new_house = pd.DataFrame(
    [[OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, FirstFlrSF, FullBath, YearBuilt]],
    columns=['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt']
)
predicted_price = best_model.predict(new_house)

print("predicted price for the new house and Condition is " , predicted_price[0],"inr")

# # Visualize actual vs predicted prices

def plot1(y_test, prediction):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.show()

def plot2(data):
    plt.hist(data['SalePrice'], bins=30)
    plt.xlabel("sale price")
    plt.ylabel("number of houses")
    plt.title("sale of price distribution")
    plt.show()

def Choices():
    print("Which graph do you want?")
    print("1 - Actual vs Predicted")
    print("2 - Sale Price Distribution")
    choice = input("Enter your choice (1 or 2): ")
    if choice == '1':
        plot1(y_test, prediction)   # no self, no object needed!
    elif choice == '2':
        plot2(data)                 # no self, no object needed!
    else:
        print("Invalid choice!")

Choices()  # just call it directly!
 

