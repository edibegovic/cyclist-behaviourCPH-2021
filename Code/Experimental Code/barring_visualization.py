import pandas as pd
import json
import cv2
import os
import jq
import velocity_calculation as vc
import math

# Init DataFrames
# ------------------------------------------

# Counter
columns = [
    "frameId",
    "timestamp",
    "counter_area",
    "ObjectClass",
    "UniqueID",
    "bearing_og",
    "countingDirection",
    "angle",
]

counter = pd.read_csv("../data/short_testing/counter_data.csv", names=columns)
counter = counter[counter.duplicated(subset=["UniqueID"], keep=False)].sort_values(
    "UniqueID"
)
counter["timestamp"] = pd.to_datetime(
    counter["timestamp"], format="%Y-%m-%dT%H:%M:%S.%fZ"
)

# Calculates speeds (and append to df)
# speeds: {UniqueID: speed}
speeds = vc.calculate_speed(2.6, counter)
for key, value in speeds.items():
    counter.loc[counter["UniqueID"] == key, "velocity"] = value

# Tracker
with open("../data/short_testing/tracker.json") as f:
    tracker = json.load(f)

tracker_flattend = pd.json_normalize(
    tracker, record_path="objects", meta=["frameId"]
).rename(columns={"id": "UniqueID"})
tracker_flattend = tracker_flattend.drop_duplicates(
    subset=["UniqueID", "frameId"], keep="first"
)

tracker_flattend[:50]

# Merge counter and tracker
# ------------------------------------------

df = pd.merge(counter, tracker_flattend, on=["UniqueID", "frameId"], how="left")

# Adding center coordinates for bounding box
# df['x_c'] = df['x'] - df['w']/2
# df['y_c'] = df['y'] - df['h']/2

# Print
df[["frameId", "UniqueID", "timestamp", "bearing_og", "angle", "x", "y", "velocity"]]

# Image Generation
# ------------------------------------------


def save_frame(frame_number, source, arrows=None):
    vc = cv2.VideoCapture(source)
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    rval, frame = vc.read()
    if arrows != None:
        for a in arrows:
            frame = cv2.arrowedLine(
                frame, a["start"], a["end"], (0, 0, 255), thickness=8, tipLength=0.6
            )
    cv2.imwrite(str(frame_number) + ".jpg", frame)


video_location = "../data/short_copy.mov"


def get_arrow(obj):
    x, y = obj["x"], obj["y"]
    angle = obj["bearing_og"]
    speed = obj["velocity"]
    a = 30 * speed * math.sin(math.radians(angle))
    b = 30 * speed * math.cos(math.radians(angle))
    return {"start": (int(x), int(y)), "end": (int(x + a), int(y + b))}


for _, row in df.iterrows():
    save_frame(row["frameId"], video_location, arrows=[get_arrow(row)])
    print(get_arrow(row))

save_frame(672, video_location)
