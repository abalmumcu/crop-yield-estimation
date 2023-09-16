import glob
import pandas as pd

class datasetloader:
    def __init__(self,dataset_folder_path):
        self.dataset_folder = dataset_folder_path
        self.datasets = []
        self.state_names = {}
        
    def load_all_pickle_dataset(self):
        all_dataset = glob.glob(self.dataset_folder+'/*.pkl')
        for itm in all_dataset:
            file_ = open(itm,'rb')
            object_file = pd.read_pickle(file_)
            self.datasets.append(object_file)
        return self.datasets

    def get_as_dict(self,dataset):
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
    
    def combine_states(self,dataset_dict,datasets,dataset_number):
        for state in dataset_dict.keys():
            for city in dataset_dict[state]:
                if city == 'baldwin':
                    dataset_first = datasets[dataset_number][state][city]['dynamic'].reset_index()
                    dataset_first['state'] = [str(state)] * len(datasets[dataset_number][state][city]['dynamic'].evi)
                    dataset_first['city'] = [str(city)] * len(datasets[dataset_number][state][city]['dynamic'].evi)
                    static_features = datasets[dataset_number][state][city]['static']
                    for i, static_feature in enumerate(static_features):
                        dataset_first[f'sf_{i+1}'] = static_feature
                    yield_data_1 = datasets[dataset_number][state][city]['yield']
                    dataset_first['yield'] = [yield_data_1] * len(datasets[dataset_number][state][city]['dynamic'].evi)


                    dataset_second = datasets[dataset_number][state]["calhoun"]['dynamic'].reset_index()
                    static_features = datasets[dataset_number][state]["calhoun"]['static']
                    dataset_second['state'] = [str(state)] * len(datasets[dataset_number][state]["calhoun"]['dynamic'].evi)
                    dataset_second['city'] = [str("calhoun")] * len(datasets[dataset_number][state]["calhoun"]['dynamic'].evi)
                    for i, static_feature in enumerate(static_features):
                        dataset_second[f'sf_{i+1}'] = static_feature
                    yield_data_2 = datasets[dataset_number][state]["calhoun"]['yield']
                    dataset_second['yield'] = [yield_data_2] * len(datasets[dataset_number][state]['calhoun']['dynamic'].evi)
                    dataset = pd.concat([dataset_first,dataset_second],ignore_index=False)
                if (city != 'baldwin') and (city != 'calhoun'):
                    dataset_tmp = datasets[dataset_number][state][city]['dynamic'].reset_index()
                    yield_data_tmp = datasets[dataset_number][state][city]['yield']
                    static_features = datasets[dataset_number][state][city]['static']

                    dataset_tmp['state'] = [str(state)] * len(datasets[dataset_number][state][city]['dynamic'].evi)
                    dataset_tmp['city'] = [str(city)] * len(datasets[dataset_number][state][city]['dynamic'].evi)
                    for i, static_feature in enumerate(static_features):
                        dataset_tmp[f'sf_{i+1}'] = static_feature
                    dataset_tmp['yield'] = [yield_data_tmp] * len(datasets[dataset_number][state][city]['dynamic'].evi)
                    dataset = pd.concat([dataset,dataset_tmp],ignore_index=False)
        return dataset
    
    def combine_datasets(self,datasets,dataset_dict,total_dataset_number):
        tmp = []
        for idx in range(total_dataset_number):
            tmp.append(self.combine_states(dataset_dict,datasets,dataset_number=idx))
        full_dataset = pd.concat(tmp,ignore_index=False)
        return full_dataset

    def first_two_state_datasets(self,datasets,dataset_number,state,city):
        dataset_first = datasets[dataset_number][state][city]['dynamic'].reset_index()
        yield_data_1 = datasets[dataset_number][state][city]['yield'].reset_index()
        dataset_first['yield'] = [yield_data_1] * len(datasets[dataset_number][state][city]['dynamic'].evi)
        static_features = datasets[dataset_number][state][city]['static'].reset_index()
        for i, static_feature in enumerate(static_features):
                dataset_first[f'sf_{i+1}'] = static_feature

        dataset_second = datasets[dataset_number][state]["calhoun"]['dynamic'].reset_index()
        yield_data_2 = datasets[dataset_number][state]["calhoun"]['yield'].reset_index()
        dataset_second['yield'] = [yield_data_2] * len(datasets[dataset_number][state]['calhoun']['dynamic'].evi)
        static_features = datasets[dataset_number][state]["calhoun"]['static'].reset_index()
        for i, static_feature in enumerate(static_features):
                dataset_second[f'sf_{i+1}'] = static_feature
        return dataset_first, dataset_second

    def other_state_datasets(self,datasets,dataset_number,state,city):
        dataset_tmp = datasets[dataset_number][state][city]['dynamic'].reset_index()
        yield_data_tmp = datasets[dataset_number][state][city]['yield'].reset_index()
        dataset_tmp['yield'] = [yield_data_tmp] * len(datasets[dataset_number][state][city]['dynamic'].evi)
        static_features = datasets[dataset_number][state][city]['static'].reset_index()
        for i, static_feature in enumerate(static_features):
                dataset_tmp[f'sf_{i+1}'] = static_feature
        return dataset_tmp
    
    def combine_states_per_12_days(self,dataset_dict,datasets,dataset_number):
        for state in dataset_dict.keys():
            for city in dataset_dict[state]:
                if city == 'baldwin':
                    if dataset_number == 1:
                        dataset_first, dataset_second = self.first_two_state_datasets(datasets,dataset_number,state,city)
                        dataset = pd.concat([dataset_first.iloc[0::12, :],dataset_second.iloc[0::12, :]],ignore_index=True)
                    elif dataset_number == 2:
                        dataset_first, dataset_second = self.first_two_state_datasets(datasets,dataset_number,state,city)
                        dataset = pd.concat([dataset_first.iloc[4::12, :],dataset_second.iloc[4::12, :]],ignore_index=True)
                    else:
                        dataset_first, dataset_second = self.first_two_state_datasets(datasets,dataset_number,state,city)
                        dataset = pd.concat([dataset_first.iloc[8::12, :],dataset_second.iloc[8::12, :]],ignore_index=True) 

                if (city != 'baldwin') and (city != 'calhoun'):
                    if dataset_number == 1:
                        dataset_tmp = self.other_state_datasets(datasets,dataset_number,state,city)
                        dataset = pd.concat([dataset,dataset_tmp.iloc[0::12, :]],ignore_index=True)

                    elif dataset_number == 2:
                        dataset_tmp = self.other_state_datasets(datasets,dataset_number,state,city)
                        dataset = pd.concat([dataset,dataset_tmp.iloc[4::12, :]],ignore_index=True)

                    else:
                        dataset_tmp = self.other_state_datasets(datasets,dataset_number,state,city)
                        dataset = pd.concat([dataset,dataset_tmp.iloc[8::12, :]],ignore_index=True)
        return dataset


    def combine_datasets_per_12_days(self,datasets,dataset_dict,total_dataset_number):
        tmp = []
        for idx in range(total_dataset_number):
            tmp.append(self.combine_states_per_12_days(dataset_dict,datasets,dataset_number=idx))
        full_dataset = pd.concat(tmp,ignore_index=True)
        return full_dataset

    
    




