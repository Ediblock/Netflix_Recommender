import re
import tensorflow as tf
import pandas as pd
import os
import numpy as np

def change_commas_to_semicolons_movie_list(file_path:str):
    with open(file_path, "r") as file:
        text = file.readline()
    if ";" in text:
        return 0
    else:
        with open(file_path, "r") as file:
            text_lines = file.readlines()
        for line_number, line in enumerate(text_lines):
            replaced_text = re.sub(r"(\d+)(?:,)(\d+)(?:,)", "\g<1>;\g<2>;", line)
            text_lines[line_number] = replaced_text
        with open(file_path, "w") as file:
            file.writelines(text_lines)

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
def create_dataframe_from_dataset(base_path_to_dataset_files: str):
    dataset = tf.data.Dataset.list_files(base_path_to_dataset_files + "/*.txt")
    size_counter = 0
    data_total = pd.DataFrame(columns=["id", "rating", "date", "movie_id"])
    data_total["movie_id"] = data_total["movie_id"].astype(np.int16)
    all_files_size = get_size(base_path_to_dataset_files)
    for file_path in dataset.as_numpy_iterator():
        file_path_decoded = file_path.decode()
        with open(file_path_decoded, "r") as file:
            movie_id = file.readline()
            movie_id = int(movie_id[0:-2])  # without last character which is ":"
        data_df = pd.read_csv(file_path_decoded, sep=",", header=None, skiprows=1, names=["id", "rating", "date"])
        data_df["movie_id"] = movie_id
        data_df["movie_id"] = data_df["movie_id"].astype(np.int16)
        data_total = pd.concat([data_total, data_df], ignore_index=True)

        size_counter += os.path.getsize(file_path_decoded)
        printProgressBar(size_counter, all_files_size)
    data_total.to_csv("./data/data_for_emb.csv", index=False)