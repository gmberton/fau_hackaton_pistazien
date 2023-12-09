
import sys
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
import lovely_tensors as lt
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import parser
import commons
import visualizations
import trained_models
from test_dataset import TestDataset

lt.monkey_patch()
args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.setup_logging(output_folder, stdout="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

model, descriptors_dimension = trained_models.get_model(args.method)
model = model.eval().to(args.device)

test_ds = TestDataset(args.database_folder, args.queries_folder)
logging.info(f"Testing on {test_ds}")

with torch.inference_mode():
    logging.debug("Extracting database descriptors for evaluation/testing")
    database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
    database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
    all_descriptors = np.empty((len(test_ds), descriptors_dimension), dtype="float32")
    for images, indices in tqdm(database_dataloader, ncols=100):
        descriptors = model(images.to(args.device))
        logging.debug(f"descriptors: {descriptors}")
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors
        
    logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
    queries_subset_ds = Subset(test_ds,
                                list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)))
    queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                    batch_size=1)
    for images, indices in tqdm(queries_dataloader, ncols=100):
        descriptors = model(images.to(args.device))
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors

queries_descriptors = all_descriptors[test_ds.database_num:]
database_descriptors = all_descriptors[:test_ds.database_num]

# Use a kNN to find predictions
faiss_index = faiss.IndexFlatL2(descriptors_dimension)
faiss_index.add(database_descriptors)
del database_descriptors, all_descriptors

logging.debug("Calculating recalls")
args.num_preds_to_save = max(args.num_preds_to_save_in_images, args.num_preds_to_save_in_excel)
_, predictions = faiss_index.search(queries_descriptors, args.num_preds_to_save)

# Save visualizations of predictions
if args.num_preds_to_save != 0:
    logging.info("Saving final predictions")
    # For each query save num_preds_to_save predictions
    visualizations.save_preds(
        predictions,
        test_ds,
        output_folder,
        args.num_preds_to_save_in_images,
        args.num_preds_to_save_in_excel,
    )

