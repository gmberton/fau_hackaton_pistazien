import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="resnet18",
        choices=["resnet18", "mixvpr", "fasternet", "mage", "clip", "g2sd"],
        help="_",
    )
    parser.add_argument(
        "--dataset", type=str, default="others",
        choices=["others", "cifar", "cxr", "stlucia"],
        help="_"
    )
    parser.add_argument("--database_folder", type=str, default="images_train", help="_")
    parser.add_argument("--queries_folder", type=str, default="images_test", help="_")
    parser.add_argument("--num_workers", type=int, default=4, help="_")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="set to 1 if database images may have different resolution",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default",
        help="experiment name, output logs will be saved under logs/exp_name",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_"
    )
    parser.add_argument(
        "--num_preds_to_save_in_excel",
        type=int,
        default=10,
        help="set != 0 if you want to save predictions for each query",
    )
    parser.add_argument(
        "--num_preds_to_save_in_images",
        type=int,
        default=3,
        help="set != 0 if you want to save predictions for each query",
    )
    
    args = parser.parse_args()
    
    return args
