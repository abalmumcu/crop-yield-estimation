import glob
import pandas as pd


class DatasetLoader:
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
