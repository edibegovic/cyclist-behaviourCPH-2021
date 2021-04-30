#!git clone https://github.com/abewley/sort.git
import pandas as pd
import numpy as np
from sort import *

def set_unique_id(df, max_age=30, min_hits=1, iou_threshold=0.15):
    tracker = Sort(max_age, min_hits, iou_threshold)

    df_grouped = df.groupby("frame_id")
    list_with_id = []

    for _, group in df_grouped:
        object_list = []
        for _, row in group.iterrows():
            object_list.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"], row["confidence"]])
        track_with_id = tracker.update(np.array(object_list))
        list_with_id.append(track_with_id)

    df_data = []
    for idx, i in enumerate(list_with_id):
        for row in i:
            x = ((row[2] - row[0])/2) + row[0]
            y = ((row[3] - row[1])/2) + row[1]
            h = (row[3] - row[1])
            w = (row[2] - row[0])
            temp = [x, y, int(row[4]), idx, "blue", h, w]
            df_data.append(temp)

    df_ids = pd.DataFrame(data=df_data, columns=["x", "y", "unique_id", "frame_id", "color", "h", "w"])
    return df_ids

if __name__ == "__main__":
    pass