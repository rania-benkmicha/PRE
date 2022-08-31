"""
Rename images from a dataset in order of increasing 'traversability' for visual analysis
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

#DATASET = "datasets/dataset_sample_bag/"
DATASET = "datasets/dataset_rania_2022-07-01-11-40-52/"

csv_file=DATASET+'/_imu_data.csv'
data_frame = pd.read_csv(csv_file)

sorted_frame = data_frame.sort_values(by=['y'])

try:
    os.mkdir("results/" + DATASET)
    print("results/" + DATASET + " folder created")
except OSError:
    print("couldn't create results/" + DATASET + " folder")
    exit()

pbar = tqdm(total=sorted_frame.shape[0])
for index, row in sorted_frame.iterrows():
    pbar.update()
    img_name = os.path.join(DATASET+'_zed_node_rgb_image_rect_color', row['image_id'])
    image = Image.open(img_name)
    image.save(f"results/{DATASET}{int(row['y']*10000):05d}.png", "PNG")

