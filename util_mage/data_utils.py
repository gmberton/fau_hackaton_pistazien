import numpy as np
import matplotlib.pyplot as plt
import yaml
import joblib
import os

#function to load the configuration file
def get_configuration_file():
  # Load the YAML file into a Python dictionary
  with open('config_PMSD.yaml', 'r') as config_file:
      config = yaml.safe_load(config_file)
  return config

#function that load a configuration file  with a specific name
def get_configuration_file_name(name):
  # Load the YAML file into a Python dictionary
  with open(name, 'r') as config_file:
      config = yaml.safe_load(config_file)
  return config

#create directory if it doesn't exist
def create_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory (including parent directories) if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

#helper functions for extracting data from a pickle file and dumping data into a pickle file
def extract_data(image_path):
  open_file = open(image_path, 'rb')
  data  = joblib.load(open_file)
  open_file.close()
  return data

def dump_data(data, image_path):
  open_file = open(image_path, 'wb')
  joblib.dump(data, open_file)
  open_file.close()

