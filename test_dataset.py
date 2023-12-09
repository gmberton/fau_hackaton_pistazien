
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd


def read_images(dir_path):
    paths = sorted(glob(f"{dir_path}/**/*.png", recursive=True)
                   + glob(f"{dir_path}/**/*.jpg", recursive=True)
                   + glob(f"{dir_path}/**/*.jpeg", recursive=True))
    return paths

class TestDataset(data.Dataset):
    def __init__(self, database_folder, queries_folder):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        
        self.database_paths = read_images(database_folder)
        self.queries_paths = read_images(queries_folder)
        
        self.images_paths = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.base_transform(pil_img)
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< #queries: {self.queries_num}; #database: {self.database_num} >"


class CXRTestDataset(data.Dataset):
    def __init__(self, data_folder, database_file, queries_file):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        data_folder : str, should contain the path to the val or test set,
            which contains all the data files.
        database_file : str, name of csv file with the names of database images.
        queries_file : str, name of csv file with the names of queries images.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        
        self.database_paths = pd.read_csv(database_file)["Image Index"].tolist()[:600]
        self.queries_paths = pd.read_csv(queries_file)["Image Index"].tolist()[50:60]

        self.database_paths = [f'{data_folder}/{p}' for p in self.database_paths]
        self.queries_paths = [f'{data_folder}/{p}' for p in self.queries_paths]
        
        self.images_paths = self.database_paths.copy()
        self.images_paths += self.queries_paths
        
        self.database_num = len(self.database_paths) 
        self.queries_num = len(self.queries_paths)
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.base_transform(pil_img)
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< #queries: {self.queries_num}; #database: {self.database_num} >"