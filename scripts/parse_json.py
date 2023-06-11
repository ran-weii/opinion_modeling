import argparse
import os
import glob
import json
import pandas as pd
from tqdm import tqdm

from src.pipeline.preprocessing import process_json_item

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--save", type=bool_, default=True)
    arglist = vars(parser.parse_args())
    return arglist

def process_json_file(raw_json):
    """ Process all items in a json file """
    out = []
    for i, (key, item) in tqdm(enumerate(raw_json.items())):
        item_dict = {"tweet_id": key}
        processed_item = process_json_item(item)
        item_dict.update(processed_item)
        out.append(item_dict) 

        if i == 100:
            break
    return out

def main(arglist):
    print("parsing json files")
    save_path = os.path.join(arglist["data_path"], "raw_csv")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_paths = glob.glob(
        os.path.join(arglist["data_path"], "raw_json/text", "*.json")
    )
    
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path).replace(".json", ".csv")
        print(f"processing {i}: {file_name}")

        with open(file_path) as f:
            raw_json = json.load(f)
        
        processed_data = process_json_file(raw_json)
        df_data = pd.DataFrame(processed_data)

        if arglist["save"]:
            df_data.to_csv(os.path.join(save_path, file_name), index=False) 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)