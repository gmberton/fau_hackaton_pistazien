
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--method", type=str, default="resnet18",
                        choices=["resnet18", "mixvpr", "clip", "g2sd"],
                        help="_")
    parser.add_argument("--database_folder", type=str, default="images_train", help="_")
    parser.add_argument("--queries_folder", type=str, default="images_test", help="_")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="_")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="set to 1 if database images may have different resolution")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="experiment name, output logs will be saved under logs/exp_name")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="_")
    parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="values for recall (e.g. recall@1, recall@5)")
    parser.add_argument("--num_preds_to_save", type=int, default=3,
                        help="set != 0 if you want to save predictions for each query")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="set to true if you want to save predictions only for "
                        "wrongly predicted queries")
    
    args = parser.parse_args()
    
    return args

