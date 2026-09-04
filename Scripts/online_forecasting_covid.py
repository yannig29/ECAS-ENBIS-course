import sys
from pathlib import Path

# Relative path definition
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_dir = project_root / "Data"
res_dir = project_root / "Results"

# add "parent" folder to python path
sys.path.append(str(project_root))

# Imports standards
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
import itertools

# Metric
from sklearn.metrics import root_mean_squared_error as rmse

#Time series
from statsmodels.tsa.arima.model import ARIMA

#Time series Additive Models
import tam as ta

# Import Viking (KALMAN)
import  viking_kalman as viking

# Aggregation of experts
import opera 

# Foundation Models
from tabicl import TabICLClassifier, TabICLRegressor
from chronos import BaseChronosPipeline, Chronos2Pipeline

#tree based regressor
import pytabkit
import lightgbm
from sklearn.model_selection import train_test_split
from pytabkit import LGBM_HPO_TPE_Regressor, LGBM_TD_Regressor
from pytabkit import RealMLP_TD_Regressor
from pytabkit import RealMLP_HPO_Regressor


# Historical data used to train the models
Data0 = pd.read_csv(data_dir / "data_train.csv")
Data0['Date'] = pd.to_datetime(Data0['Date'])
Data0.insert(2, "WeekDays2", Data0['Date'].dt.dayofweek, True)
n0 = len(Data0)
print(n0)

# Test data used to evaluate the forecasts
Data1 = pd.read_csv(data_dir /'data_test.csv')
Data1['Date'] = pd.to_datetime(Data1['Date'])
Data1.insert(2, "WeekDays2", Data1['Date'].dt.dayofweek, True)
n1 = len(Data1)
print(n1)

### Complete Dataset for the Kalman filter
Data = pd.concat([Data0, Data1], ignore_index=True)
d1 = Data[Data['Date'].astype(str) == "2020-03-15"].index[0]

###covariate types
cat_cols = ['WeekDays']
num_cols = ['Load.1', 'Load.7', 'Temp', 'Temp_s95',
            'Temp_s99', 'Temp_s95_min', 'Temp_s95_max', 'Temp_s99_min',
            'Temp_s99_max','GovernmentResponseIndex', 'toy', 'Time', 'WeekDays2',
            'BH', 'DLS', 'Summer_break', 'Christmas_break']
target_col = 'Load'

Data0[cat_cols] = Data0[cat_cols].astype('category')
Data1[cat_cols] = Data1[cat_cols].astype('category')

# 
X_train = Data0[cat_cols + num_cols]
y_train =  Data0[[target_col]]
X_test = Data1[cat_cols + num_cols]





#########################################################################################################
################ Descriptive Analysis
#########################################################################################################

#Electricity consumption (Load)
plt.figure(figsize=(15, 6))
plt.plot(Data0['Date'], Data0['Load'], label='Train', color='#1f77b4', linewidth=1.2)
plt.plot(Data1['Date'], Data1['Load'], label='Test', color='#ff7f0e', linewidth=1.5)
plt.xlabel('Date', fontsize=11, fontweight='bold', color='#555555')
plt.ylabel('Load', fontsize=11, fontweight='bold', color='#555555')
plt.title('Electricity Consumption (Train and Test)', fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.grid(axis='x', linestyle=':', alpha=0.3)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
plt.legend(loc='upper left', frameon=True, edgecolor='#cccccc', facecolor='white', framealpha=1)
plt.tight_layout()
plt.show()



#Distribution of Load
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(
    Data0['Load'], 
    bins=100, 
    color="steelblue", 
    kde=True,          # Adds a smooth Kernel Density Estimate line
    edgecolor="white", # Separates the bins visually
    alpha=0.9          # Slight transparency
)
plt.xlabel('Load', fontsize=12, fontweight='bold', color='#333333')
plt.ylabel('Frequency', fontsize=12, fontweight='bold', color='#333333')
plt.title('Distribution of Load', fontsize=14, fontweight='bold', pad=15)
sns.despine()
plt.tight_layout()
plt.show()

# Yearly Cycle
a, b = 0, 365
plt.figure(figsize=(10, 6))
plt.plot(Data0['Date'].iloc[a:b], Data0['Load'].iloc[a:b])
plt.xlabel('Date')
plt.ylabel('Load')
plt.title('Yearly Cycle of Load')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='toy', y='Load', data=Data0, alpha=0.2)
plt.xlabel('Time of Year')
plt.ylabel('Load')
plt.title('Load vs Time of Year')
plt.show()


# Weekly Cycle
plt.figure(figsize=(10, 6))
plt.plot(Data0.loc[ (Data0['Month'] == 6) & (Data0['Year']==2018), 'Load'])
plt.xlabel('Day of June')
plt.ylabel('Load')
plt.title('Weekly Cycle in June')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='WeekDays', y='Load', data=Data0)
plt.xlabel('WeekDays')
plt.ylabel('Load')
plt.title('Load by WeekDays')
plt.show()


# Time dependance
fig, ax = plt.subplots(1,2,figsize=(10,5))
plot_acf(Data0['Load'], lags=70, ax=ax[0])
plot_pacf(Data0['Load'], lags=70, ax=ax[1])
plt.show()


# Scatter plot Load/Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temp', y='Load', data=Data0, alpha=0.3, color='orchid')
#sns.scatterplot(x='Temp_s95', y='Load', data=Data0, alpha=0.3, color='blue')
plt.xlabel('Temperature')
plt.ylabel('Load')
plt.title('Load vs Temperature')
plt.show()


# Train Test Distribution: Temperature
plt.figure(figsize=(10, 6))
plt.hist(Data0['Temp'], bins=50, color='blue', label='Train', density=True, histtype='step', linewidth=2)
plt.hist(Data1['Temp'], bins=50, color='red', label='Test', density=True, histtype='step', linewidth=2)
plt.xlabel('Temperature')
plt.ylabel('Density') # Changed from Frequency since density=True
plt.title('Train vs Test Temperature Distribution')
plt.legend() 
plt.show()

# Train Test Distribution: GVI
plt.figure(figsize=(10, 6))
plt.plot(Data0['Date'], Data0['GovernmentResponseIndex'], label='Train')
plt.plot(Data1['Date'], Data1['GovernmentResponseIndex'], label='Test')
plt.xlabel('Date')
plt.ylabel('Government Response Index')
plt.title('Government Response Index over Time')
plt.show()




#########################################################################################################
################ TAM step by step
#########################################################################################################
features = cat_cols + num_cols
cols_ws = ["Date", "Load"] + features

#####univariate TAM
formula_tam = (
        "Load ~ "       
        "s(Temp, k=20, deg=3, p=2, ap=-30)"                                             
    )

model_lambda1 = ta.StaticTAM(formula=formula_tam, date_col='Date')
model_lambda1.fit(Data0[cols_ws])
model_lambda1_train = model_lambda1.predict(Data0[cols_ws])["EstimatedLoad"]
model_lambda1_prediction = model_lambda1.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data0['Load'], model_lambda1_train))
print(rmse( Data1['Load'], model_lambda1_prediction))


formula_tam = (
        "Load ~ "       
        "s(Temp, k=20, deg=3, p=2, ap=0)"                                             
    )
cols_ws = ["Date", "Load"] + features
model_lambda2 = ta.StaticTAM(formula=formula_tam, date_col='Date')
model_lambda2.fit(Data0[cols_ws])
model_lambda2_train = model_lambda2.predict(Data0[cols_ws])["EstimatedLoad"]
model_lambda2_prediction = model_lambda2.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data0['Load'], model_lambda2_train))
print(rmse( Data1['Load'], model_lambda2_prediction))


####extrapolation and plot
grid_df = pd.DataFrame({
    'Temp': np.linspace(-10, 40, 500)
})

grid_df['Date'] = pd.to_datetime('2022-01-01')
grid_df['Load'] = 0.0
for col in features:
    if col != 'Temp':
        grid_df[col] = 0.0

effect_lambda1 = model_lambda1.predict(grid_df)["EstimatedLoad"]
effect_lambda2 = model_lambda2.predict(grid_df)["EstimatedLoad"]

plt.figure(figsize=(10, 6))
plt.scatter(Data0['Temp'], Data0['Load'], facecolor='gray', edgecolors='none', alpha=0.1, label='Actual Train Data')
plt.plot(grid_df['Temp'], effect_lambda1, color='red', linewidth=2.5, label="ap=-30 (Under-penalized / Wiggly)")
plt.plot(grid_df['Temp'], effect_lambda2, color='blue', linewidth=2.5, label="ap=0 (Regularized / Smooth)")

plt.title('Effect of Regularization on Temperature (spline) Effect ')
plt.xlabel('Temperature (°C)')
plt.ylabel('Estimated Load')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.show()


############minimising the GCV on a grid
formula_auto = (
    "Load ~ "
    "s(Temp, k=20, deg=3, p=2)"
)

model_auto = ta.StaticTAM(formula=formula_auto, date_col='Date')

model_auto.auto_fit(Data0[cols_ws], alpha_p_list=np.linspace(-30, 10, 100))
model_auto.summary()
model_auto_prediction = model_auto.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data1['Load'], model_auto_prediction))


##############################
######Model 1
##############################
formula_tam = (
        "Load ~ "       
        "s(Temp, k=20, deg=3, p=2, ap=0) + "                            
        "s(toy, k=20, deg=3, p=2, ap=-5) + "    
        "l(Time, ap=-5)"                                     
    )

model1 = ta.StaticTAM(formula=formula_tam, date_col='Date')

start = time.time()
model1.fit(Data0[cols_ws])
time_fit = time.time() - start

model1_train = model1.predict(Data0[cols_ws])["EstimatedLoad"]
model1_prediction = model1.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data1['Load'], model1_prediction))
model1.summary()



def plot_tam_effect(model, original_data, feature_name, feature_list, range_min, range_max, num_points=500, ax=None):
    """
    Automates the extraction and plotting of a GAM partial effect from a TAM model.
    Accepts an optional 'ax' parameter for subplots.
    """
    # 1 & 2. Construction de la grille (inchangé)
    grid_df = pd.DataFrame({feature_name: np.linspace(range_min, range_max, num_points)})
    grid_df['Date'] = pd.to_datetime('2022-01-01')
    grid_df['Load'] = 0.0
    for col in feature_list:
        if col != feature_name:
            grid_df[col] = 0.0
            
    float_cols = grid_df.select_dtypes(include=['float']).columns
    grid_df[float_cols] = grid_df[float_cols].astype(np.float32)
    
    # 3. Prédiction
    effect = model.predict(grid_df)["EstimatedLoad"]
    
    # 4. Tracé dynamique (gère le cas où on l'appelle seule ou dans un subplot)
    show_plot_at_end = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        show_plot_at_end = True
        
    ax.plot(grid_df[feature_name], effect, color='blue', linewidth=2.5)
    ax.scatter(original_data[feature_name], 
               np.full(len(original_data), effect.min()), 
               alpha=0.05, color='gray', marker='|')
               
    ax.set_title(f'Effect: {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Contribution to Load')
    ax.grid(True, alpha=0.3)
    
    if show_plot_at_end:
        plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
plot_tam_effect(model1, Data0, feature_name='Temp', feature_list=features, range_min=min(Data0['Temp']), range_max=max(Data0['Temp']), num_points=500, ax=axes[0])
plot_tam_effect(model1, Data0, feature_name='toy', feature_list=features, range_min=min(Data0['toy']), range_max=max(Data0['toy']), num_points=500, ax=axes[1])
plot_tam_effect(model1, Data0, feature_name='Time', feature_list=features, range_min=min(Data0['Time']), range_max= max(Data0['Time']), num_points=500, ax=axes[2])
plt.tight_layout() 
plt.show()


###autofit

formula_auto = (
    "Load ~ "       
        "s(Temp, k=20, deg=3, p=2) + "                            
        "s(toy, k=20, deg=3, p=2) + "    
        "l(Time, ap=-5)"          
)

model_auto = ta.StaticTAM(formula=formula_auto, date_col='Date')
model_auto.auto_fit(Data0[cols_ws], alpha_p_list=np.linspace(-30, 10, 10))
model_auto.summary()
model_auto_prediction = model_auto.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data1['Load'], model_auto_prediction))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
plot_tam_effect(model_auto, Data0, feature_name='Temp', feature_list=features, range_min=min(Data0['Temp']), range_max=max(Data0['Temp']), num_points=500, ax=axes[0])
plot_tam_effect(model_auto, Data0, feature_name='toy', feature_list=features, range_min=min(Data0['toy']), range_max=max(Data0['toy']), num_points=500, ax=axes[1])
plot_tam_effect(model_auto, Data0, feature_name='Time', feature_list=features, range_min=min(Data0['Time']), range_max= max(Data0['Time']), num_points=500, ax=axes[2])
plt.tight_layout() 
plt.show()




#residuals analysis
residuals = Data0['Load'] - model1_train

plt.figure(figsize=(15, 5))
plt.plot(Data0['Date'], residuals, linewidth=0.3, alpha=0.8, color='#1f77b4')
plt.title('Residuals over Time')
plt.ylabel('Residuals (Load)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.show()

sns.boxplot(x = Data0['WeekDays2'], y = residuals)





##############################
######Model 2
##############################
formula_tam = (
        "Load ~ "       
        "s(Temp, k=20, deg=3, p=2, ap=-5) + "                            
        "s(toy, k=20, deg=3, p=2, ap=-5) + "    
        "l(Time, ap=-5) + "     
        "c(WeekDays2, topo='nominal', ap=-5)"                                      
    )


model2 = ta.StaticTAM(formula=formula_tam, date_col='Date')

start = time.time()
model2.fit(Data0[cols_ws])
time_fit = time.time() - start

model2_train = model2.predict(Data0[cols_ws])["EstimatedLoad"]
model2_prediction = model2.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data1['Load'], model2_prediction))
model2.summary()

residuals = Data0['Load'] - model2_train
plt.figure(figsize=(15, 5))
plt.plot(Data0['Date'], residuals, linewidth=0.3, alpha=0.8, color='#1f77b4')
plt.title('Residuals over Time')
plt.ylabel('Residuals (Load)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.show()


# Time dependance
fig, ax = plt.subplots(1,2,figsize=(10,5))
plot_acf(residuals, lags=70, ax=ax[0])
plot_pacf(residuals, lags=70, ax=ax[1])
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(Data0['Load.1'], residuals, s=5, alpha=0.2, color='#1f77b4')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title('Residuals')
plt.xlabel('Load1 (lag 1)')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)
plt.show()




##############################
######Model 3
##############################

formula_tam = (
        "Load ~ "       
        "s(Temp, k=20, deg=3, p=2, ap=-5) + "                            
        "s(toy, k=20, deg=3, p=2, ap=-5) + "    
        "s(Time, k=5, ap=-1) +"   
        "c(WeekDays2, topo='nominal', ap=-5) +" 
        "l(Load.1, ap=-5)+"
        "l(Load.7, ap=-5)"                                   
    )



model3 = ta.StaticTAM(formula=formula_tam, date_col='Date')

start = time.time()
model3.fit(Data0[cols_ws])
time_fit = time.time() - start

model3_train = model3.predict(Data0[cols_ws])["EstimatedLoad"]
model3_prediction = model3.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data1['Load'], model3_prediction))
model3.summary()

residuals = Data0['Load'] - model3_train
plt.figure(figsize=(15, 5))
plt.plot(Data0['Date'], residuals, linewidth=0.3, alpha=0.8, color='#1f77b4')
plt.title('Residuals over Time')
plt.ylabel('Residuals (Load)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.show()


# Time dependance
fig, ax = plt.subplots(1,2,figsize=(10,5))
plot_acf(residuals, lags=70, ax=ax[0])
plot_pacf(residuals, lags=70, ax=ax[1])
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(Data0['Load.1'], residuals, s=5, alpha=0.2, color='#1f77b4')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title('Residuals')
plt.xlabel('Load1 (lag 1)')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)
plt.show()


#COVID
plt.figure(figsize=(10, 6))
plt.scatter(Data0['GovernmentResponseIndex'], residuals, s=5, alpha=0.2, color='#1f77b4')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title('Residuals')
plt.xlabel('GRI')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)
plt.show()


##############################
######Model 4
##############################
formula_tam = (
        "Load ~ "       
        "s(Temp, k=20, deg=3, p=2, ap=-5) + "                            
        "s(toy, k=20, deg=3, p=2, ap=-5) + "    
        "s(Time, k=5, ap=-1) +"   
        "c(WeekDays2, topo='nominal', ap=-5, n_cat=7) +" 
        "l(Load.1, ap=-5) +"
        "l(Load.7, ap=-5) +"
        "c(BH, topo='nominal', ap=-5, n_cat=2) +"      
        "c(Summer_break, topo='nominal', ap=-5, n_cat=2) +" 
        "c(Christmas_break, topo='nominal', ap=-5, n_cat=2)"                                  
    )


        #"c(GovernmentResponseIndex, topo='nominal', ap=10)"       
model4 = ta.StaticTAM(formula=formula_tam, date_col='Date')

start = time.time()
model4.fit(Data0[cols_ws])
time_fit = time.time() - start

model4_train = model4.predict(Data0[cols_ws])["EstimatedLoad"]
model4_prediction = model4.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data1['Load'], model4_prediction))
model4.summary()

residuals = Data0['Load'] - model4_train
plt.figure(figsize=(15, 5))
plt.plot(Data0['Date'], residuals, linewidth=0.3, alpha=0.8, color='#1f77b4')
plt.title('Residuals over Time')
plt.ylabel('Residuals (Load)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
axes_flat = axes.flatten()
plot_tam_effect(model4, Data0, feature_name='Temp', feature_list=features, range_min=min(Data0['Temp']), range_max=max(Data0['Temp']), num_points=500, ax=axes[0])
plot_tam_effect(model4, Data0, feature_name='toy', feature_list=features, range_min=min(Data0['toy']), range_max=max(Data0['toy']), num_points=500, ax=axes[1])
plot_tam_effect(model4, Data0, feature_name='Time', feature_list=features, range_min=min(Data0['Time']), range_max= max(Data0['Time']), num_points=500, ax=axes[2])
plot_tam_effect(model4, Data0, feature_name='Load.1', feature_list=features, range_min=min(Data0['Load']), range_max=max(Data0['Load']), num_points=500, ax=axes_flat[3])
plt.tight_layout() 
plt.show()



#####exemple of a neural additive model

formula_nam = (
        "Load ~ "       
        "n(Temp, n_neurons = 30, n_hidden_layers =1) + "  
        #"te(n(Time, n_neurons = 30, n_hidden_layers =1), n(Temp, n_neurons = 30, n_hidden_layers =1))+"                              
        "n(toy, n_neurons = 30, n_hidden_layers =3) + "    
        "n(Time, n_neurons = 30, n_hidden_layers =1, ap=-1) +"   
        "c(WeekDays2, topo='nominal', ap=-5, n_cat=7) +" 
        "l(Load.1, ap=-5) +"
        "l(Load.7, ap=-5) +"
        "c(BH, topo='nominal', ap=-5, n_cat=2) +"      
        "c(Summer_break, topo='nominal', ap=-5, n_cat=2) +" 
        "c(Christmas_break, topo='nominal', ap=-5, n_cat=2)"                                  
    )


        #"c(GovernmentResponseIndex, topo='nominal', ap=10)"       
model5 = ta.StaticTAM(formula=formula_nam, date_col='Date')

start = time.time()
model5.fit(Data0[cols_ws])
time_fit = time.time() - start

model5_train = model5.predict(Data0[cols_ws])["EstimatedLoad"]
model5_prediction = model5.predict(Data1[cols_ws])["EstimatedLoad"]
print(rmse( Data1['Load'], model5_prediction))
model5.summary()

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
axes_flat = axes.flatten()
plot_tam_effect(model5, Data0, feature_name='Temp', feature_list=features, range_min=min(Data0['Temp']), range_max=max(Data0['Temp']), num_points=500, ax=axes[0])
plot_tam_effect(model5, Data0, feature_name='toy', feature_list=features, range_min=min(Data0['toy']), range_max=max(Data0['toy']), num_points=500, ax=axes[1])
plot_tam_effect(model5, Data0, feature_name='Time', feature_list=features, range_min=min(Data0['Time']), range_max= max(Data0['Time']), num_points=500, ax=axes[2])
plot_tam_effect(model5, Data0, feature_name='Load.1', feature_list=features, range_min=min(Data0['Load']), range_max=max(Data0['Load']), num_points=500, ax=axes_flat[3])
plt.tight_layout() 
plt.show()



########################################################################################################################
#Error correction, time varying ARIMA
########################################################################################################################
res0 = np.ravel(Data0['Load']-model4_train)
res1 = np.ravel(Data1['Load']-model4_prediction)


#plot de l'erreur du modèle fixe
plt.plot(Data0['Date'], res0, label='Train')
plt.plot(Data1['Date'], res1, label='Test', color='red')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residuals over Time')
plt.legend()
plt.show()


model = ARIMA(res0, order=(7,0,0))
model_fit = model.fit()
output = model_fit.forecast()

###online ARIMA fitted on a the n last values of the residuals
history = list(res0)
predictions=list()
n = 7*20  #window length

for t in tqdm(range(len(res1)), desc="Processing data"):
    window_history = history[-n:] if len(history) - n > 0 else history
    model = ARIMA(window_history, order=(7, 0, 0))
    model_fit = model.fit(method='burg')
    output = model_fit.forecast()
    yhat = output
    predictions.append(yhat[0])
    obs = res1[t]
    history.append(obs)

tam_arima =  model4_prediction + predictions

score = rmse( Data1['Load'], tam_arima)
print(f"#########################################TAM ARIMA RMSE: {score:.4f}")


plt.figure(figsize=(10, 6))
res2 = res1-predictions
plt.plot(Data1['Date'], res1, label='GAM residuals', color='red')
plt.plot(Data1['Date'], res2, label='GAM+AR', color='purple')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residuals over Time')
plt.legend()
plt.show()


########################################################################################################################
#Adaptation: Kalman
########################################################################################################################

########################################################
####2 dimensional Kalman TAM
########################################################
#Data = pd.concat([Data0, Data1], ignore_index=True)

TAM_prediction = model4.predict(Data)["EstimatedLoad"]
y = Data['Load'].to_numpy()
intercept = np.ones((len(y), 1)) 
Xkal = np.column_stack((intercept, TAM_prediction))

# means = np.mean(Xkal, axis=0)
# stds = np.std(Xkal, axis=0)
# stds[stds == 0] = 1.0
# Xkal = (Xkal - means) / stds


#### static Kalman
ssm = viking.statespace.StateSpaceModel(Xkal, y, kalman_params= None)
print(ssm)
ssm.kalman_params
pred_TAM_static = ssm.pred_mean[d1+1:]
score = rmse(y[d1+1:], pred_TAM_static)
print(f"#########################Static 2-dim TAM RMSE: {score:.4f}")


#### dynamic Kalman (grid search)
l = viking.iterative_grid_search(Xkal[0:d1+1,:], y[0:d1+1], q_list =  2.0 ** np.arange(-50, 1), p1 = 1, ncores=10, max_iter=100)  
ssm_dyn = viking.statespace.StateSpaceModel(Xkal, y, kalman_params= l)
pred_TAM_dyn_em = ssm_dyn.pred_mean[d1+1:]
score = rmse(y[d1+1:], pred_TAM_dyn_em)
print(f"#########################Dyn grid search 2-dim TAM RMSE: {score:.4f}")

#### dynamic Kalman (EM)
l_em = viking.expectation_maximization(Xkal[0:d1+1,:], y[0:d1+1], Q_init = np.eye(Xkal.shape[1]), p1 = 1, n_iter = 10)
ssm_dyn_em = viking.statespace.StateSpaceModel(Xkal, y, kalman_params= l_em)
pred_TAM_dyn_em = ssm_dyn_em.pred_mean[d1+1:]
score = rmse(y[d1+1:], pred_TAM_dyn_em)
print(f"############################Dynamic EM 2-dim TAM RMSE: {score:.4f}")


####kalman p-dimensional
formula_tam = (
        "Load ~ "       
        "s(Temp, k=20, deg=3, p=2, ap=-5) + "                            
        "s(toy, k=20, deg=3, p=2, ap=-5) + "    
        "s(Time, k=5, ap=-1) +"   
        "c(WeekDays2, topo='nominal', ap=-5, n_cat=7) +" 
        "l(Load.1, ap=-5) +"
        "l(Load.7, ap=-5) +"
        "c(BH, topo='nominal', ap=-5, n_cat=2) +"      
        "c(Summer_break, topo='nominal', ap=-5, n_cat=2) +" 
        "c(Christmas_break, topo='nominal', ap=-5, n_cat=2)"                                  
    )




def extract_normalized_effects(model, data, feature_list, d1):
    """
    Extrait l'effet partiel de chaque terme additif du modèle TAM sur un jeu de données.
    
    Args:
        model: Le modèle TAM entraîné.
        data: Le DataFrame contenant les données historiques.
        feature_list: La liste stricte des variables prédictives.
        
    Returns:
        Une matrice NumPy 2D (lignes = observations, colonnes = effets partiels normalisés).
        L'ordre des colonnes correspond exactement à l'ordre de `feature_list`.
    """
    # 1. Initialisation d'une matrice vide (lignes = taille des données, colonnes = nb de features)
    # On utilise float32 pour rester cohérent et optimisé avec ton backend MPS
    n_rows = len(data)
    n_cols = len(feature_list)
    effects_matrix = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    for i, feature in enumerate(feature_list):
        # Créer une copie isolée des données réelles
        temp_data = data.copy()
        
        # Neutraliser TOUTES les autres variables prédictives
        for col in feature_list:
            if col != feature:
                if pd.api.types.is_integer_dtype(temp_data[col]):
                    temp_data[col] = 0
                else:
                    temp_data[col] = 0.0
                    
        # Prédire la charge avec cette seule variable active
        raw_effect = model.predict(temp_data)["EstimatedLoad"]
        
        # Convertir en tableau NumPy (au cas où la prédiction soit une Series Pandas)
        if isinstance(raw_effect, pd.Series):
            raw_effect = raw_effect.to_numpy()
            
        # Normalisation 
        normalized_effect = (raw_effect - raw_effect[0:d1+1].mean())/raw_effect[0:d1+1].std()
        
        # Stocker le résultat dans la i-ème colonne de la matrice
        effects_matrix[:, i] = normalized_effect
        
    return effects_matrix
feature_list = [
    'Temp',
    'toy',
    'Time',
    'WeekDays2',
    'Load.1',
    'Load.7',
    'BH',
    'Summer_break',
    'Christmas_break'
]




Xkal2 = extract_normalized_effects(model=model4, data=Data, feature_list=feature_list, d1=d1)
Xkal2 = np.column_stack((intercept,Xkal2))
Xkal2.shape

#### static Kalman
ssm2 = viking.statespace.StateSpaceModel(Xkal2, y, kalman_params= None)
pred_TAM_static2 = ssm2.pred_mean[d1+1:]
score = rmse(y[d1+1:], pred_TAM_static2)
print(f"#########################################Static p-dim TAM RMSE: {score:.4f}")


#### dynamic Kalman (EM)
l_em = viking.expectation_maximization(Xkal2[0:d1+1,:], y[0:d1+1], Q_init = np.eye(Xkal2.shape[1]), p1 = 1, n_iter = 10)
ssm_dyn_em2 = viking.statespace.StateSpaceModel(Xkal2, y, kalman_params= l_em)
pred_TAM_dyn_em2 = ssm_dyn_em2.pred_mean[d1+1:]
score = rmse(y[d1+1:], pred_TAM_dyn_em2)
print(f"#########################################Dynamic P-dim TAM RMSE: {score:.4f}")


#### dynamic scalar
def create_kalman_experts(rates, Xkal, y, d1):
    """
    Generates Kalman filter predictions for a list of Q-matrix scaling rates.
    Returns a DataFrame containing all the new expert predictions.
    """
    experts_dict = {}
    y_test = y[d1+1:]
    p = Xkal.shape[1]  # The dimension of your state space

    print(f"Generating Kalman experts for {len(rates)} different rates...")

    for rate in rates:
        # 1. Build the scaled diagonal Q matrix
        Q = np.eye(p) * rate

        # 2. Run the EM initialization (n_iter = 0)
        l_big = viking.expectation_maximization(
            Xkal,
            y,
            Q_init=Q,
            p1=1,
            n_iter=0
        )

        # 3. Fit the State Space Model
        ssm_dyn_big = viking.statespace.StateSpaceModel(Xkal, y, kalman_params=l_big)

        # 4. Extract the predictions for the test set
        pred = ssm_dyn_big.pred_mean[d1+1:]

        # 5. Calculate RMSE to see how this specific rate performed
        score = rmse(y_test, pred)
        print(f"Rate {rate:<5} -> RMSE: {score:.4f}")

        # 6. Store it in the dictionary with a clean name
        expert_name = f"kalman_Q_{rate}"
        experts_dict[expert_name] = pred

    # Convert everything into a DataFrame
    experts_df_new = pd.DataFrame(experts_dict)

    print("\n Returned DataFrame with shape:", experts_df_new.shape)
    return experts_df_new

my_rates = [0.001, 0.01,0.1, 1,10]
new_kalman_experts_df = create_kalman_experts(my_rates, Xkal2, y, d1)

new_kalman_experts_df.columns
agg_kalman = opera.Mixture(
    y=y[d1+1:],
    experts=new_kalman_experts_df,
    model="MLpol",
    loss_type="mse",
    loss_gradient=True,
)

rmse(y[d1+1:], agg_kalman.predictions)
score = rmse(Data1['Load'], agg_kalman.predictions)
print(f"#########################################Agg Kalman GAM RMSE: {score:.4f}")

agg_kalman.plot_mixture()


def plot_cumulative_errors(dates, y, ychap, metric='squared'):
    y_array = np.asarray(y).reshape(-1, 1)
    
    if isinstance(ychap, pd.DataFrame):
        experts = ychap.values
        labels = ychap.columns
    else:
        experts = np.asarray(ychap)
        labels = [f"Expert {i}" for i in range(experts.shape[1])]
        
    if metric == 'squared':
        errors = (experts - y_array) ** 2
        y_label = "Cumulative Squared Error"
    elif metric == 'absolute':
        errors = np.abs(experts - y_array)
        y_label = "Cumulative Absolute Error"
    else:
        raise ValueError("metric has to be either 'squared' or 'absolute'")
        
    cum_errors = np.cumsum(errors, axis=0)
    
    # Configuration du graphique
    plt.figure(figsize=(12, 6))
    
    for i in range(cum_errors.shape[1]):
        plt.plot(dates, cum_errors[:, i], label=labels[i], linewidth=1.5)
        
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.title("Cumulative Errors")
    
    # Placement de la légende à l'extérieur pour ne pas cacher les courbes
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() 
    
    plt.show()

prev_mat=new_kalman_experts_df
prev_mat['aggregation'] = agg_kalman.predictions

plot_cumulative_errors(Data1['Date'], Data1['Load'], prev_mat, metric='squared')






########################################################################################################################
#Online Chronos, exemple sur 3 jours de prévision, puis chargement des prév. calculer préalablement.
########################################################################################################################
Data0['Item_id'] = 'France'
Data1['Item_id'] = 'France'

#####Cronos Config
target = "Load"  # Column name containing the values to forecast (elec consumption)
prediction_length = Data1.shape[0]  # Number of hours to forecast ahead
id_column = "Item_id"  # Column identifying different time series (countries/regions)
timestamp_column = "Date"  # Column containing datetime information
Future_cov = Data1.drop(columns=['Load', 'Load.1', 'Load.7'])
Future_cov.columns

###################################################################################################
#########Online cronos
###################################################################################################
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="mps")
s = np.arange(0, Data1.shape[0])

# Initialisation
pred_init = pipeline.predict_df(
    Data0,
    future_df=Future_cov.iloc[[0]],
    prediction_length=1,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target
)

list_preds = [pred_init]
len_data0 = len(Data0)
full_data = pd.concat([Data0, Data1], axis=0)

#loop for online forecasting online
#replace s[0:2] with s[:-1] to simulate online forecast on all the test set
#warning: long time, arround 26 minutes on my laptop with a GPU

for i in tqdm(s[:-1], desc="Processing data"):
#for i in tqdm(s[0:2], desc="Processing data"):
    context_df = full_data.iloc[:len_data0 + i + 1]
    future_df = Future_cov.iloc[[(i+1)]]
    
    pred = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=1,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column=id_column,
        timestamp_column=timestamp_column,
        target=target,
        validate_inputs=False
    )
    list_preds.append(pred)

pred_df_ol = pd.concat(list_preds, axis=0).reset_index(drop=True)
pred_df_ol.shape

# with open(f'{res_dir}/Expe_covid1/pred_Chronos_ol.pkl', "wb") as f:
#         pickle.dump(pred_df_ol, f)

with open(f'{res_dir}/Expe_covid1/pred_Chronos_ol.pkl', "rb") as f:
     pred_chronos_ol = pickle.load(f)

pred_chronos_ol = pred_chronos_ol['predictions']
score = rmse(Data1['Load'], pred_chronos_ol)
print(f"################Online Chronos RMSE: {score:.4f}")




###################################################################################################
#########Online cronos, window
###################################################################################################
# Define your window size in terms of number of observations
#
window_steps = 30  
window_steps = 30*3
window_steps = 365  
window_steps = 2*365 
window_steps = 3*365 

s = np.arange(0, Data1.shape[0])

# 1. Initialisation (also restricted to the window size)
context_init = Data0.iloc[-window_steps:] if len(Data0) > window_steps else Data0

pred_init = pipeline.predict_df(
    context_init,
    future_df=Future_cov.iloc[[0]],
    prediction_length=1,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target
)

list_preds = [pred_init]
len_data0 = len(Data0)
full_data = pd.concat([Data0, Data1], axis=0)

# 2. Loop for online forecasting with a rolling window
for i in tqdm(s[:-1], desc="Processing data"):
    
    # Calculate the dynamic start and end indices
    end_idx = len_data0 + i + 1
    start_idx = max(0, end_idx - window_steps)
    
    # Slice only the recent 'window_steps' rows
    context_df = full_data.iloc[start_idx : end_idx]
    future_df = Future_cov.iloc[[(i+1)]]
    
    pred = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=1,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column=id_column,
        timestamp_column=timestamp_column,
        target=target,
        validate_inputs=False
    )
    list_preds.append(pred)

# 3. Concatenate and evaluate
pred_df_ol = pd.concat(list_preds, axis=0).reset_index(drop=True)
print(f"Shape of predictions: {pred_df_ol.shape}")

# with open(f'{res_dir}/Expe_covid1/pred_Chronos_ol_w{window_steps}.pkl', "wb") as f:
#         pickle.dump(pred_df_ol, f)

with open(f'{res_dir}/Expe_covid1/pred_Chronos_ol_w{window_steps}.pkl', "rb") as f:
      pred_chronos_ol_w = pickle.load(f)

pred_chronos_ol_w = pred_chronos_ol_w['predictions'] 
score = rmse(Data1['Load'], pred_chronos_ol_w)
print(f"######################Online Chronos RMSE: {score:.4f}")




###################################################################################################
#########Chronos aggregation
###################################################################################################
window_steps=30
filepath = f'{res_dir}/Expe_covid1/pred_Chronos_ol_w{window_steps}.pkl'
with open(filepath, "rb") as f:
    pred_chronos_ol_w_30 = pickle.load(f) # Nom de variable standard

window_steps=90
filepath = f'{res_dir}/Expe_covid1/pred_Chronos_ol_w{window_steps}.pkl'
with open(filepath, "rb") as f:
    pred_chronos_ol_w_90 = pickle.load(f) # Nom de variable standard

window_steps=365
filepath = f'{res_dir}/Expe_covid1/pred_Chronos_ol_w{window_steps}.pkl'
with open(filepath, "rb") as f:
    pred_chronos_ol_w_365 = pickle.load(f) # Nom de variable standard

window_steps=730
filepath = f'{res_dir}/Expe_covid1/pred_Chronos_ol_w{window_steps}.pkl'
with open(filepath, "rb") as f:
    pred_chronos_ol_w_730 = pickle.load(f) # Nom de variable standard

score = rmse(Data1['Load'], pred_chronos_ol_w_730['predictions'])
print(f"######################Online Chronos RMSE: {score:.4f}")

window_steps=1095
filepath = f'{res_dir}/Expe_covid1/pred_Chronos_ol_w{window_steps}.pkl'
with open(filepath, "rb") as f:
    pred_chronos_ol_w_1095 = pickle.load(f) # Nom de variable standard

score = rmse(Data1['Load'], pred_chronos_ol_w_1095['predictions'])
print(f"######################Online Chronos RMSE: {score:.4f}")


experts_chronos_df = pd.DataFrame({
    'Chronos_ol': pred_chronos_ol,
    'Chronos_w_30': pred_chronos_ol_w_30['predictions'],
    'Chronos_w_90': pred_chronos_ol_w_90['predictions'],
    'Chronos_w_365': pred_chronos_ol_w_365['predictions'],
    'Chronos_w_730': pred_chronos_ol_w_730['predictions'],
    'Chronos_w_1095': pred_chronos_ol_w_1095['predictions']
})

# experts_chronos_df = pd.DataFrame({
#         'Chronos_w_1095_01' : pred_chronos_ol_w_1095['0.1'],
#         'Chronos_w_1095_05' : pred_chronos_ol_w_1095['0.5'],
#         'Chronos_w_1095_09' : pred_chronos_ol_w_1095['0.9']
#     })

agg_chronos = opera.Mixture(
    y=y[d1+1:],
    experts=experts_chronos_df,
    model="MLpol",
    loss_type="mse",
    loss_gradient=True,
)

score = rmse(Data1['Load'], agg_chronos.predictions)
print(f"#########################################Agg Chronos RMSE: {score:.4f}")

agg_chronos.plot_mixture()


rmse_scores = {
    expert: rmse(Data1['Load'], experts_chronos_df[expert].values)
    for expert in experts_chronos_df.columns
}

rmse_scores


########################################################################################################################
#TabICL.
########################################################################################################################
X_train = Data0[cat_cols + num_cols].copy()
y_train =  Data0[[target_col]].copy()
X_test = Data1[cat_cols + num_cols].copy()

reg = TabICLRegressor( verbose=True, random_state=30, n_estimators=10)
reg.fit(X_train, y_train.values.ravel())
pred_tabICL = reg.predict(X_test)

score = rmse(Data1['Load'], pred_tabICL)
print(f"#########################################OFFline TabICL RMSE: {score:.4f}")



########################################################################################################################
#Kalman TabICL.
########################################################################################################################
X = pd.concat([X_train, X_test],ignore_index=True)
pred_tabICL_all = reg.predict(X)
y = Data['Load'].to_numpy()

intercept = np.ones(len(y))
intercept = intercept.reshape(-1, 1)
Xkal = np.column_stack((intercept,pred_tabICL_all))

#### static Kalman
ssm = viking.statespace.StateSpaceModel(Xkal, y, kalman_params= None)
print(ssm)
tabICL_static = ssm.pred_mean[d1+1:]
score = rmse(Data1['Load'], tabICL_static)
print(f"#########################Static TabICL RMSE: {score:.4f}")

#### dynamic Kalman (EM)
l_em = viking.expectation_maximization(Xkal[0:d1+1,:], y[0:d1+1], Q_init = np.eye(Xkal.shape[1]), p1 = 1, n_iter = 10)  #######1.8s
ssm_dyn_em = viking.StateSpaceModel(Xkal, y, kalman_params= l_em)
tabICL_dyn_em = ssm_dyn_em.pred_mean[d1+1:]
score = rmse(Data1['Load'], tabICL_dyn_em)
print(f"##########################Dynamic TabICL RMSE: {score:.4f}")




############################################################################################################
############Agregation
############################################################################################################

experts = {
    'TAM': model4_prediction,
    'TAM_arima': tam_arima,
    'TAM_kalman': pred_TAM_dyn_em,
    'TAM_kalman2': pred_TAM_dyn_em2,
    'Agg_kalman': agg_kalman.predictions,
    'chronos': pred_chronos_ol,
    'agg_chronos':  agg_chronos.predictions,
    'tabICL': pred_tabICL,
    'tabICL_dyn_em': tabICL_dyn_em,
}

experts_flat = {key: np.ravel(value) for key, value in experts.items()}
experts_df = pd.DataFrame(experts_flat)


agg= opera.Mixture(
    y=y[d1+1:],
    experts=experts_df,
    model="MLpol",
    #model="MLProd",
    loss_type="mse",
    loss_gradient=True,
)
agg.plot_mixture()
score = rmse(Data1['Load'], agg.predictions)
print(f"#########################################Online Aggregation RMSE: {score:.4f}")





########################################################################################################################
#Search for the best aggregation a posteriori
########################################################################################################################
y_test = y[d1+1:]
all_experts = list(experts_df.columns)

best_overall_rmse = float('inf')
best_ensemble = []

# Optional: Calculate total combinations just so you know what to expect
total_combinations = (2 ** len(all_experts)) - 1
print(f"Starting exhaustive search across {total_combinations} combinations...\n")

# Loop through all possible ensemble sizes (from 1 expert up to all experts)
for r in range(1, len(all_experts) + 1):

    # Generate all combinations of size 'r'
    for subset in itertools.combinations(all_experts, r):
        test_group = list(subset)

        # Fit the mixture model on this specific combination
        agg = opera.Mixture(
            y=y_test,
            experts=experts_df[test_group],
            model="MLpol",
            loss_type="mse",
            loss_gradient=True,
        )

        ensemble_pred = agg.predictions
        score = rmse(y_test, ensemble_pred)

        # If this combination is the best we've seen so far, save it!
        if score < best_overall_rmse:
            best_overall_rmse = score
            best_ensemble = test_group
            print(f"New best found! RMSE: {best_overall_rmse:.4f} | Experts: {best_ensemble}")

# Create the final dataframe with the absolute best combination
final_experts_df = experts_df[best_ensemble]

print("\n" + "="*50)
print(f"Final chosen experts: {best_ensemble}")
print(f"Final Best RMSE: {best_overall_rmse:.4f}")
print("="*50)




########################################################################################################################
#other tabular  methods
########################################################################################################################
np.random.seed(0)
X = pd.concat([X_train, X_test],ignore_index=True)
y = Data['Load'].to_numpy()

model = RealMLP_TD_Regressor()  # tuned defaults
model.fit(X_train, y_train)
pred_RealMLP = model.predict(X_test)
score = rmse(Data1['Load'], pred_RealMLP)
print(f"#########################################realMLP tuned default RMSE: {score:.4f}")


model2 = RealMLP_HPO_Regressor(n_hyperopt_steps=2)  #  hyper parameter optimisation
model2.fit(X_train, y_train)
pred_RealMLP2 = model2.predict(X_test)
score = rmse(Data1['Load'], pred_RealMLP2)
print(f"#########################################realMLP HPO RMSE: {score:.4f}")



lgbm_HPO = LGBM_HPO_TPE_Regressor(val_fraction=0.9, n_hyperopt_steps=5, random_state=0)
lgbm_HPO.fit(X_train, y_train)
pred_lgbm_HPO = lgbm_HPO.predict(X_test)
score = rmse(Data1['Load'], pred_lgbm_HPO)
print(f"#########################################lgbm HPO RMSE: {score:.4f}")






y = Data['Load'].to_numpy()
intercept = np.ones((len(y), 1)) 

pred = lgbm_HPO.predict(X)
Xkal = np.column_stack(intercept, pred)

#### static Kalman
ssm = viking.statespace.StateSpaceModel(Xkal, y, kalman_params= None)
print(ssm)
ssm.kalman_params
pred_lgbm_static = ssm.pred_mean[d1+1:]
score = rmse(y[d1+1:], pred_lgbm_static)
print(f"#########################Static 2-dim TAM RMSE: {score:.4f}")

#### dynamic Kalman (grid search)
l = viking.iterative_grid_search(Xkal[0:d1+1,:], y[0:d1+1], q_list =  2.0 ** np.arange(-50, 1), p1 = 1, ncores=10, max_iter=100)  
ssm_dyn = viking.statespace.StateSpaceModel(Xkal, y, kalman_params= l)
pred_TAM_dyn_em = ssm_dyn.pred_mean[d1+1:]
score = rmse(y[d1+1:], pred_TAM_dyn_em)
print(f"#########################Dyn grid search 2-dim TAM RMSE: {score:.4f}")

#### dynamic Kalman (EM)
l_em = viking.expectation_maximization(Xkal[0:d1+1,:], y[0:d1+1], Q_init = np.eye(Xkal.shape[1]), p1 = 1, n_iter = 10)
ssm_dyn_em = viking.statespace.StateSpaceModel(Xkal, y, kalman_params= l_em)
pred_lgbm_dyn_em = ssm_dyn_em.pred_mean[d1+1:]
score = rmse(y[d1+1:], pred_lgbm_dyn_em)
print(f"############################Dynamic EM 2-dim TAM RMSE: {score:.4f}")




