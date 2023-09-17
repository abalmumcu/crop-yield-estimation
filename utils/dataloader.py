import glob
import pandas as pd
import numpy as np

class DatasetLoader:
    def __init__(self,dataset_folder_path):
        self.dataset_folder = dataset_folder_path
        self.datasets = []
        self.state_names = {}
        self.selected_day = 15
        self.window_size = 7   # Length of the moving average filter
        self.static_feature_names = ["Sand", "Silt", "Clay",
                                    "Bulk density",
                                    "Coarse fragments",
                                    "Total Nitrogen", "pH", 
                                    "CEC", "SOC", "OCD", "OCS"] 
        
    def load_all_pickle_dataset(self):
        all_dataset = glob.glob(self.dataset_folder+'/*.pkl')
        for itm in all_dataset:
            file_ = open(itm,'rb')
            object_file = pd.read_pickle(file_)
            self.datasets.append(object_file)
        return self.datasets

    def get_state_names(self,dataset):
        states = []
        for key, _ in dataset.items():
            states.append(key)
        cities = []
        for state in states:
            tmp = []
            for key, _ in dataset[state].items():
                tmp.append(key)
            cities.append(tmp)
        for idx in range(len(states)):
            self.state_names[states[idx]] = cities[idx]
        return self.state_names
    
    def data_preprocessing(self,datasets, dataset_number,state, city):
        df = datasets[dataset_number][state][city]['dynamic']
        # df['state'] = [str(state)] * len(datasets[dataset_number][state][city]['dynamic'].evi)
        # df['city'] = [str(city)] * len(datasets[dataset_number][state][city]['dynamic'].evi)
        static_features = datasets[dataset_number][state][city]['static']
        for col, static_feature in enumerate(static_features):
            df[self.static_feature_names[col]] = static_feature.astype(np.float64)
                        
        yield_data_1 = datasets[dataset_number][state][city]['yield']
        df['yield'] = [yield_data_1] * len(datasets[dataset_number][state][city]['dynamic'].evi)

        df.index = pd.to_datetime(df.index)
        df_all = df.sort_index(ascending=True)
        
        # interpolation
        df_all = df.interpolate()
        df_monthly = df_all[(df_all.index.month >= 4) & (df_all.index.month <= 9)]

        filtered_df = self.moving_average_filter(df_monthly)
        filtered_df.set_index(df_monthly.index, inplace=True)

        df_dayly = filtered_df[(filtered_df.index.day == self.selected_day)]

        df_dayly.index = pd.to_datetime(df_dayly.index, format = '%Y-%m-%d').strftime('%m-%d')

        for col in range(1,13):
            df_dayly[df_dayly.columns[col]] = df_dayly[df_dayly.columns[col]].astype('float64')

        if all(0 <= x <= 1 for x in df_dayly.iloc[:, 0]):
            df_filtered = df_dayly.select_dtypes(include=['float64', 'int64'])  

        return df_filtered.reset_index()


    def moving_average_filter(self,df_monthly):
        filter_radius = self.window_size // 2  # Radius of the filter (number of days before and after the current day)
        filtered_df = pd.DataFrame()  # DataFrame to store the filtered data

        # Apply moving average filter to each feature
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
        
    def combine_states(self,dataset_dict,datasets,dataset_number):
        for state in dataset_dict.keys():
            for city in dataset_dict[state]:
                if city == 'baldwin':
                    dataset_first = self.data_preprocessing(datasets, dataset_number,state, city)
                    dataset_second = self.data_preprocessing(datasets, dataset_number,state, city="calhoun")
                    dataset = pd.concat([dataset_first,dataset_second],ignore_index=False)

                if (city != 'baldwin') and (city != 'calhoun'):
                    dataset_tmp = self.data_preprocessing(datasets, dataset_number,state, city)
                    dataset = pd.concat([dataset,dataset_tmp],ignore_index=False)
        return dataset
    
    def combine_datasets(self,datasets,dataset_dict,total_dataset_number):
        tmp = []
        for idx in range(total_dataset_number):
            tmp.append(self.combine_states(dataset_dict,datasets,dataset_number=idx))
        full_dataset = pd.concat(tmp,ignore_index=False)

        full_dataset = full_dataset.reset_index(drop=True)
        full_dataset.rename(columns={'index':'Date'}, inplace=True)
        return full_dataset