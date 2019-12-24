import os
import random

dataset_dir = "datasets"

with open(os.path.join(dataset_dir, "total.txt"), "r") as f:
    lines = f.readlines()
    random.shuffle(lines)
    num = int(len(lines) * 0.7)
    with open(os.path.join(dataset_dir, "train.txt"), "w") as train_file:
        for line in lines[:num]:
            train_file.write(line)
    with open(os.path.join(dataset_dir, "test.txt"), "w") as test_file:
        for line in lines[num:]:
            test_file.write(line)