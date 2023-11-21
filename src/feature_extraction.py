from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import seaborn as sns 
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
from utils.dataloader import DatasetLoader 
from utils.plot import plt_corr

import warnings
warnings.filterwarnings('ignore')

dataset_path = "dataset/"
loader = DatasetLoader(dataset_path)

datasets = loader.load_all_pickle_dataset()
dataset_dict = loader.get_state_names(datasets[0])
print(dataset_dict)

blacklist_states = ['kansas', 'new mexico', 'california', 'arizona']

for idx in range(len(blacklist_states)):
    del dataset_dict[blacklist_states[idx]]

full_df = loader.combine_datasets(datasets,dataset_dict,total_dataset_number=5)

full_df['Date'] = pd.to_datetime(full_df['Date'], format = '%m-%d')

cols = list(full_df.columns)
dynamic_cols = cols[1:14]
new_dynamic_cols = [f"{dyn}_{m}" for dyn in dynamic_cols for m in range(4, 10)]

vector_df = pd.DataFrame(columns=new_dynamic_cols)
mean_df = pd.DataFrame(columns=cols[1:])

for i in range(0, len(full_df), 6):
    row = full_df.iloc[i:i+6,1:14].values.ravel()
    vector_df = vector_df.append(pd.Series(row, index=vector_df.columns), ignore_index=True)
    mean_ = []
    for j in range(1, len(cols)):
        values = full_df.iloc[i:i+6, j].values
        mean_.append(values.mean())
    mean_df = mean_df.append(pd.Series(mean_, index=cols[1:]), ignore_index=True)

for i in range(13,len(cols)):
    vector_df[cols[i]] = mean_df[cols[i]]

plt_corr(mean_df,size=(30,30),save_fig=True, save_path= "results/corr_map.png")

# XGBoost

target = 'yield'

#df_no_data_removed_ = full_df.drop('Date',axis=1)


X = mean_df.drop(target,axis=1)

y = mean_df[target]

#  convert the dataset into an optimized data structure called Dmatrix that XGBoost supports 
data_dmatrix = xgb.DMatrix(data=X,label=y) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41) 

# fit model no training data
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror') 
xg_reg.fit(X_train,y_train) 
# plot feature importance

xgb.plot_importance(xg_reg) 
plt.rcParams['figure.figsize'] = [15, 15] 
plt.show()

plt.barh(mean_df.columns[:-1], xg_reg.feature_importances_)

sorted_idx = xg_reg.feature_importances_.argsort()
plt.barh(mean_df.columns[:-1][sorted_idx], xg_reg.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")


# SHAP

# Fits the explainer
explainer = shap.TreeExplainer(xg_reg)
# Calculates the SHAP values - It takes some time
_shap_values = explainer.shap_values(X_test)
shap.summary_plot(_shap_values, X_test, plot_type="bar",max_display=30)

shap.summary_plot(_shap_values, X_test,max_display=30)


feature_names = X_train.columns
vals = np.abs(_shap_values).mean(0)

rf_resultX = pd.DataFrame(_shap_values, columns = X.columns)

vals = np.abs(rf_resultX).mean(0)

shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                  columns=['col_name','feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)
shap_importance.head(25)

# Permutation Method

perm_importance = permutation_importance(xg_reg, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(mean_df.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")


# HeatMap

def correlation_heatmap(train):
    correlations = train.corr()

    _, _ = plt.subplots(figsize=(20,20))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    plt.show();
    
correlation_heatmap(X_train[mean_df.columns[sorted_idx]])