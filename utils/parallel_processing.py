import numpy as np
import pandas as pd
import multiprocessing

class ParallelDataProcessing:
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
    
    def parallel_data_preprocessing(self, dataset_dict, datasets ,dataset_number, selected_day):
        results = []
        for state in dataset_dict.keys():
            for city in dataset_dict[state]:
                result = multiprocessing.Process(target=self.data_preprocessing,
                                        args =(datasets, dataset_number,state, city, selected_day))

                results.append(result)
        for j in results:
            j.start()
        for j in results:
            j.join()
        return results

    def combine_datasets_parallel(self, datasets, dataset_dict, total_dataset_number,selected_day=15):
        tmp = []
        for dataset_number in range(total_dataset_number):
            tmp.append(self.parallel_data_preprocessing(dataset_dict, datasets ,dataset_number,selected_day))
        full_dataset = pd.concat(tmp, ignore_index=True)

        full_dataset = full_dataset.reset_index(drop=True)
        full_dataset.rename(columns={'index': 'Date'}, inplace=True)
        return full_dataset