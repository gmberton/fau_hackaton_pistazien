
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.transform import rescale
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import logging
from sklearn.neighbors import KNeighborsClassifier


# Height and width of a single image
H = 512
W = 512
TEXT_H = 175
FONTSIZE = 80
SPACE = 50  # Space between two images


def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with vertical text, spaced along rows."""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE)
    img = Image.new('RGB', ((W * len(labels)) + 50 * (len(labels)-1), TEXT_H), (1, 1, 1))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        _, _, w, h = d.textbbox((0,0), text, font=font)
        d.text(((W+SPACE)*i + W//2 - w//2, 1), text, fill=(0, 0, 0), font=font)
    return np.array(img)


def draw(img, c=(0, 255, 0), thickness=20):
    """Draw a colored (usually red or green) box around an image."""
    p = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    for i in range(3):
        cv2.line(img, (p[i, 0], p[i, 1]), (p[i+1, 0], p[i+1, 1]), c, thickness=thickness*2)
    return cv2.line(img, (p[3, 0], p[3, 1]), (p[0, 0], p[0, 1]), c, thickness=thickness*2)


def build_prediction_image(images_paths, preds_correct=None):
    """Build a row of images, where the first is the query and the rest are predictions.
    For each image, if is_correct then draw a green/red box.
    """
    if preds_correct is None:
        preds_correct = [None for _ in images_paths]
    assert len(images_paths) == len(preds_correct)
    # labels = ["Query"] + [f"Pred {i} - {is_correct}" for i, is_correct in enumerate(preds_correct[1:])]
    labels = ["Query"] + [f"Pred {i}" for i, is_correct in enumerate(preds_correct[1:])]
    num_images = len(images_paths)
    images = [np.array(Image.open(path).convert("RGB")) for path in images_paths]
    for img, correct in zip(images, preds_correct):
        if correct is None:
            continue
        color = (0, 255, 0) if correct else (255, 0, 0)
        draw(img, color)
    concat_image = np.ones([H, (num_images*W)+((num_images-1)*SPACE), 3])
    rescaleds = [rescale(i, [min(H/i.shape[0], W/i.shape[1]), min(H/i.shape[0], W/i.shape[1]), 1]) for i in images]
    for i, image in enumerate(rescaleds):
        pad_width = (W - image.shape[1] + 1) // 2
        pad_height = (H - image.shape[0] + 1) // 2
        image = np.pad(image, [[pad_height, pad_height], [pad_width, pad_width], [0, 0]], constant_values=1)[:H, :W]
        concat_image[: , i*(W+SPACE) : i*(W+SPACE)+W] = image
    try:
        labels_image = write_labels_to_image(labels)
        final_image = np.concatenate([labels_image, concat_image])
    except OSError:  # Handle error in case of missing PIL ImageFont
        final_image = concat_image
    final_image = Image.fromarray((final_image*255).astype(np.uint8))
    return final_image


def save_preds(dataset, predictions, eval_ds, output_folder,
               num_preds_to_save_in_images, num_preds_to_save_in_excel,
    ):
    """For each query, save an image containing the query and its predictions,
    and a file with the paths of the query, its predictions and its positives.

    Parameters
    ----------
    predictions : np.array of shape [num_queries x num_preds_to_viz], with the preds
        for each query
    eval_ds : TestDataset
    output_folder : str / Path with the path to save the predictions
    """
    # positives_per_query = eval_ds.get_positives()
    os.makedirs(f"{output_folder}/preds", exist_ok=True)
    output_file_content = "Query,"
    for i in range(len(predictions[0])):
        output_file_content += f"Prediction {i},"
    output_file_content = output_file_content[:-1]  # remove final ","
    output_file_content += "\n"
    
    for query_index, preds in enumerate(tqdm(predictions, ncols=80, desc=f"Saving preds in {output_folder}")):
        query_path = eval_ds.queries_paths[query_index]
        list_of_images_paths = [query_path]
        # List of None (query), True (correct preds) or False (wrong preds)
        for pred_index, pred in enumerate(preds):
            pred_path = eval_ds.database_paths[pred]
            list_of_images_paths.append(pred_path)
            
            if num_preds_to_save_in_images != 0:
                prediction_image = build_prediction_image(list_of_images_paths[:num_preds_to_save_in_images + 1])
                pred_image_path = f"{output_folder}/preds/{query_index:03d}.jpg"
                prediction_image.save(pred_image_path)
        
        output_file_content += ",".join(list_of_images_paths[:num_preds_to_save_in_excel + 1])
        output_file_content += "\n"
    
    with open(f"{output_folder}/output.csv", "w") as file:
        _ = file.write(output_file_content)
    
    if dataset == "cxr":
        path_df = pd.read_csv(f'{output_folder}/output.csv')
        labels_df = pd.read_csv("/mnt/nas/Data_WholeBody/NIH_ChestX-ray8/Data_Entry_2017_v2020.csv")
        # Create a dictionary mapping from image file names to labels
        labels_dict = dict(zip(labels_df['Image Index'], labels_df['Finding Labels']))

        # Function to extract filename from path and get corresponding label
        def get_label_from_path(path):
            filename = path.split('/')[-1]
            return labels_dict.get(filename, 'Label Not Found')

        # Apply the function to each prediction column
        for col in path_df.columns[:]:
            path_df[col] = path_df[col].apply(get_label_from_path)

        # Save the updated dataframe
        path_df.to_csv(f'{output_folder}/preds_with_labels.csv', index=False)

        jac_sim = calculate_jaccard_similarity(path_df)
        print(f'Jaccard Similarity: {jac_sim:.4f}')
        logging.info(f'Jaccard Similarity: {jac_sim:.4f}')


def jaccard_similarity(set1, set2):
    """Calculate the Jaccard Similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def calculate_jaccard_similarity(df):
    jaccard_scores = []

    for index, row in df.iterrows():
        query_labels = set(row['Query'].split('|'))
        prediction_scores = []
        
        for col in df.columns[1:]:  # Skip the first column which is 'Query'
            prediction_labels = set(row[col].split('|'))
            score = jaccard_similarity(query_labels, prediction_labels)
            prediction_scores.append(score)
        
        # Average Jaccard Similarity for this row
        jaccard_scores.append(sum(prediction_scores) / len(prediction_scores))

    # Calculate overall Jaccard Similarity
    overall_jaccard_similarity = sum(jaccard_scores) / len(jaccard_scores)
    return overall_jaccard_similarity