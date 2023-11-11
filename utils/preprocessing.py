import numpy as np
import pandas as pd

class DataProcessing:
    def __init__(self,):
        self.window_size = 7   # Length of the moving average filter
        self.static_feature_names = ["Sand", "Silt", "Clay",
                                    "Bulk density",
                                    "Coarse fragments",
                                    "Total Nitrogen", "pH", 
                                    "CEC", "SOC", "OCD", "OCS"] 

   
    def data_preprocessing(self, datasets, dataset_number, state, city, selected_day):
        df = datasets[dataset_number][state][city]['dynamic']
        static_features = datasets[dataset_number][state][city]['static']
        for col, static_feature in enumerate(static_features):
            df[self.static_feature_names[col]] = static_feature.astype(np.float64)
                        
        yield_data_1 = datasets[dataset_number][state][city]['yield']
        df['yield'] = [yield_data_1] * len(datasets[dataset_number][state][city]['dynamic'].evi)

        df.index = pd.to_datetime(df.index)
        df_all = df.sort_index(ascending=True)
        
        df_all = df.interpolate()
        df_monthly = df_all[(df_all.index.month >= 4) & (df_all.index.month <= 9)]

        filtered_df = self.moving_average_filter(df_monthly)
        filtered_df.set_index(df_monthly.index, inplace=True)

        df_dayly = filtered_df[(filtered_df.index.day == selected_day)]

        df_dayly.index = pd.to_datetime(df_dayly.index, format = '%Y-%m-%d').strftime('%m-%d')
        df_dayly.iloc[:, 1:13] = df_dayly.iloc[:, 1:13].astype('float64')

        if all(0 <= x <= 1 for x in df_dayly.iloc[:, 0]):
            df_filtered = df_dayly.select_dtypes(include=['float64', 'int64'])  

        return df_filtered.reset_index()

    def moving_average_filter(self,df_monthly):
        filter_radius = self.window_size // 2  
        filtered_df = pd.DataFrame()  

        for feature in df_monthly.columns:
            moving_averages = []
            for i in range(len(df_monthly)):
                start_idx = max(i - filter_radius, 0)
                end_idx = min(i + filter_radius + 1, len(df_monthly))
                window = df_monthly.iloc[start_idx:end_idx][feature]
                window_average = round(np.mean(window), 6)
                moving_averages.append(window_average)

            filtered_df[feature] = moving_averages
        return filtered_df
        
    def combine_states(self,dataset_dict,datasets,dataset_number,selected_day):
        for state in dataset_dict.keys():
            for city in dataset_dict[state]:
                if city == 'baldwin':
                    dataset_first = self.data_preprocessing(datasets, dataset_number,state, city, selected_day)
                    dataset_second = self.data_preprocessing(datasets, dataset_number,state,selected_day = selected_day, city="calhoun",)
                    dataset = pd.concat([dataset_first,dataset_second],ignore_index=False)

                if (city != 'baldwin') and (city != 'calhoun'):
                    dataset_tmp = self.data_preprocessing(datasets, dataset_number,state, city, selected_day)
                    dataset = pd.concat([dataset,dataset_tmp],ignore_index=False)
        return dataset
    
    def combine_datasets(self,datasets,dataset_dict,total_dataset_number,selected_day = 15):
        tmp = []
        for idx in range(total_dataset_number):
            tmp.append(self.combine_states(dataset_dict,datasets,selected_day = selected_day ,dataset_number=idx))
        full_dataset = pd.concat(tmp,ignore_index=False)

        full_dataset = full_dataset.reset_index(drop=True)
        full_dataset.rename(columns={'index':'Date'}, inplace=True)
        return full_dataset
  

    def vectorize_dataset(self,full_df,dynamic_column_number):
        cols = list(full_df.columns)
        dynamic_cols = cols[1:dynamic_column_number]
        new_dynamic_cols = [f"{dyn}_{m}" for dyn in dynamic_cols for m in range(4, 10)]

        vector_df = pd.DataFrame(columns=new_dynamic_cols)
        mean_df = pd.DataFrame(columns=cols[1:])

        for i in range(0, len(full_df), 6):
            row = full_df.iloc[i:i+6,1:dynamic_column_number].values.ravel()
            vector_df = vector_df.append(pd.Series(row, index=vector_df.columns), ignore_index=True)
            mean_ = []
            for j in range(1, len(cols)):
                values = full_df.iloc[i:i+6, j].values
                mean_.append(values.mean())
            mean_df = mean_df.append(pd.Series(mean_, index=cols[1:]), ignore_index=True)

        for i in range(dynamic_column_number,len(cols)):
            vector_df[cols[i]] = mean_df[cols[i]]
        
        return vector_df, mean_df
    
    def train_test_split_wscaler(self,target,df_vector):
        from sklearn.model_selection import train_test_split 
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = []
        train_test = []
        np_train = df_vector.drop(target,axis=1).to_numpy()
        np_test = df_vector[target].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(np_train, np_test, test_size=0.20, random_state=42)

        scalerx = MinMaxScaler()
        scalerx.fit(X_train)
        X_train_minmax_scaled= scalerx.transform(X_train)
        X_test_minmax_scaled = scalerx.transform(X_test)
        
        scalery = MinMaxScaler()
        scalery.fit(y_train.reshape(-1, 1))
        y_train_minmax_scaled = scalery.transform(y_train.reshape(-1, 1))
        y_test_minmax_scaled = scalery.transform(y_test.reshape(-1, 1))
        test_minmax_scaled = scalery.transform(np_test.reshape(-1, 1))
        
        scaler.append([scalerx, scalery])

        train_test.append([X_train_minmax_scaled,X_test_minmax_scaled,y_train_minmax_scaled,y_test_minmax_scaled,test_minmax_scaled])
        
        return scaler,train_test
    
    def drop_feature(self,df,drop_feature_list):
        return df.drop(drop_feature_list,axis=1)

    def check_na(self,df):
        nan_check = df.isna().any()
        count_nan_columns = nan_check.sum()
        if count_nan_columns != 0:
            raise Exception(f"There is a NaN value in the DataFrame!\nNumber of columns with NaN values: {count_nan_columns}")
