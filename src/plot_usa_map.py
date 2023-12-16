from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from utils.plot import visualize_cities_on_map
from utils.dataloader import DatasetLoader

loader = DatasetLoader("data/dataset/")

datasets = loader.load_all_pickle_dataset()
dataset_dict = loader.get_state_names(datasets[0])

blacklist_states = ['kansas', 'new mexico', 'california', 'arizona']

for key in blacklist_states:
    dataset_dict.pop(key, None)

visualize_cities_on_map(dataset_dict)
