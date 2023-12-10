python3 main.py --dataset_folder '/mnt/nas/Data_WholeBody/NIH_ChestX-ray8/images' \
    --database_file '/mnt/nas/Data_WholeBody/NIH_ChestX-ray8/train.csv' \
    --queries_file '/mnt/nas/Data_WholeBody/NIH_ChestX-ray8/test.csv' \
    --dataset 'cxr' \
    --method 'mage' \
    --batch_size 1 \
    --num_preds_to_save 3