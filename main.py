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
from util_mage.data_utils import extract_data, dump_data
import os

import lovely_tensors as lt
import parser
import commons
import visualizations
import trained_models
from test_dataset import TestDataset, CXRTestDataset

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

# test_ds = TestDataset(args.database_folder, args.queries_folder)
if args.dataset == "cxr":
    test_ds = CXRTestDataset(args.dataset_folder, args.database_file, args.queries_file)
else:
    test_ds = TestDataset(args.database_folder, args.queries_folder)
logging.info(f"Testing on {test_ds}")

with torch.inference_mode():
    if args.dataset != "cxr":
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
    else:
        if not os.path.exists(f"database_descriptors.pkl"):
            logging.debug("Extracting database descriptors for evaluation/testing")
            database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
            database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                            batch_size=args.batch_size)
            all_descriptors = np.empty((len(test_ds), descriptors_dimension), dtype="float32")
            database_labels = []
            for images, indices, labels in tqdm(database_dataloader, ncols=100):
                descriptors = model(images.to(args.device))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors
                labels_list = [label.item() for label in labels]
                database_labels.append(labels_list)
                
            logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
            queries_subset_ds = Subset(test_ds,
                                        list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)))
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                            batch_size=1)
            queries_labels = []
            if args.dataset == "cxr":
                for images, indices, labels in tqdm(queries_dataloader, ncols=100):
                    descriptors = model(images.to(args.device))
                    descriptors = descriptors.cpu().numpy()
                    all_descriptors[indices.numpy(), :] = descriptors
                    labels_list = [label.item() for label in labels]
                    queries_labels.append(labels_list)
            else:
                for images, indices in tqdm(queries_dataloader, ncols=100):
                    descriptors = model(images.to(args.device))
                    descriptors = descriptors.cpu().numpy()
                    all_descriptors[indices.numpy(), :] = descriptors
    
            queries_descriptors = all_descriptors[test_ds.database_num:]
            database_descriptors = all_descriptors[:test_ds.database_num]
    
            dump_data(database_descriptors, f"database_descriptors.pkl")
            dump_data(queries_descriptors, f"queries_descriptors.pkl")
            if args.dataset == "cxr":
                dump_data(database_labels, f"database_labels.pkl")
                dump_data(queries_labels, f"queries_labels.pkl")
        else:
            logging.debug("Loading descriptors from disk")
            database_descriptors = extract_data(f"database_descriptors.pkl")
            queries_descriptors = extract_data(f"queries_descriptors.pkl")

# Use a kNN to find predictions
faiss_index = faiss.IndexFlatL2(descriptors_dimension)
faiss_index.add(database_descriptors)
# del database_descriptors, all_descriptors
# del all_descriptors

logging.debug("Calculating recalls")
args.num_preds_to_save = max(args.num_preds_to_save_in_images, args.num_preds_to_save_in_excel)
_, predictions = faiss_index.search(queries_descriptors, args.num_preds_to_save)

# Save visualizations of predictions
if args.num_preds_to_save != 0:
    logging.info("Saving final predictions")
    # For each query save num_preds_to_save predictions
    visualizations.save_preds(
        args.dataset,
        predictions,
        test_ds,
        output_folder,
        args.num_preds_to_save_in_images,
        args.num_preds_to_save_in_excel,
    )

if args.dataset == "cifar":
    from scripts import eval_cifar
    top = 5
    accuracy = eval_cifar.compute_accuracy(csv_path=output_folder + "/output.csv", top=top)
    logging.info(f"Top-{top} accuracy {accuracy * 100:.1f}")
elif args.dataset == "stlucia":
    from scripts import eval_stlucia
    top = 5
    accuracy = eval_stlucia.compute_accuracy(csv_path=output_folder + "/output.csv", top=top)
    logging.info(f"Top-{top} accuracy {accuracy * 100:.1f}")
if args.dataset == "cxr":
    # use logistic regression to predict the probability of each class
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import accuracy_score

    LG_model = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=999, max_iter=1000))

    # Train the model
    LG_model.fit(database_descriptors, database_labels)

    # Predict on the test set
    queries_pred = LG_model.predict(queries_descriptors)
    queries_pred = (queries_pred > 0.3).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(queries_labels, queries_pred)
    print(f'Accuracy: {accuracy}')
    logging.info(f'Logistic Regression Accuracy on CXR: {accuracy}')

    pred_score = LG_model.score(queries_descriptors, queries_labels)
    print(f'Prediction Score: {pred_score}')

