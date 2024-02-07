"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor


# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')
# Feature has no significance so its removed
train_df = train.drop(['Unnamed: 0'], axis=1)
#Replace the null value with median

train_df = train_df.drop(['Unnamed: 0'], axis=1)
train_df['Valencia_wind_deg'] = train_df['Valencia_wind_deg'].astype(str).str.extract('(\d+)', expand=False).astype(float)
train_df['Seville_pressure'] = train_df['Seville_pressure'].astype(str).str.extract('(\d+)', expand=False).astype(float)


# # Create a copy
# Train = train_df.copy()

# Train['time'] = pd.to_datetime(Train['time'])

# Train['Day'] = Train['time'].dt.day
# Train['Month'] = Train['time'].dt.month
# Train['Year'] = Train['time'].dt.year
# Train['Hour'] = Train['time'].dt.hour

# # predict_features = Train[['Year', 'Month', 'Day', 'Hour', 'Madrid_wind_speed',
# #     'Madrid_clouds_all', 'Madrid_pressure', 'Madrid_weather_id',
# #     'Seville_humidity', 'Seville_clouds_all', 'Seville_wind_speed',
# #     'Seville_pressure', 'Seville_weather_id', 'Barcelona_wind_speed',
# #     'Barcelona_wind_deg', 'Barcelona_pressure', 'Barcelona_weather_id',
# #     'Valencia_wind_speed', 'Valencia_wind_deg', 'Valencia_humidity',
# #     'Valencia_pressure', 'Bilbao_wind_speed', 'Bilbao_wind_deg',
# #     'Bilbao_clouds_all', 'Bilbao_pressure', 'Bilbao_weather_id']]

# ## Splitting our data into dependent Variable and Independent Variable
# # X_train = Train[['Year', 'Month', 'Day', 'Hour', 'Madrid_wind_speed',
# #     'Madrid_clouds_all', 'Madrid_pressure', 'Madrid_weather_id',
# #     'Seville_humidity', 'Seville_clouds_all', 'Seville_wind_speed',
# #     'Seville_pressure', 'Seville_weather_id', 'Barcelona_wind_speed',
# #     'Barcelona_wind_deg', 'Barcelona_pressure', 'Barcelona_weather_id',
# #     'Valencia_wind_speed', 'Valencia_wind_deg', 'Valencia_humidity',
# #     'Valencia_pressure', 'Bilbao_wind_speed', 'Bilbao_wind_deg',
# #     'Bilbao_clouds_all', 'Bilbao_pressure', 'Bilbao_weather_id']]

X_train =train_df.drop(columns=['load_shortfall_3h'], axis=1)

y_train = train['load_shortfall_3h']

# Fit model
gb_model = GradientBoostingRegressor(n_estimators=140, max_depth=6, random_state=42)

print ("Training Model...")
gb_model.fit(X_train, y_train)

# Pickle model for use within our API
save_path = './trained/gb_model_final.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(gb_model, open(save_path,'wb'))
