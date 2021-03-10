
import pandas as pd
import json
import cv2
import os
import jq

columns = ["frameId", "timestamp", "counter_area", "ObjectClass", 
        "UniqueID", "bearing_og", "CountingDirection", "Angle"]

counter = pd.read_csv("../data/short_testing/counter_data.csv", names=columns)

with open('../data/short_testing/tracker.json') as f:
    tracker = json.load(f)

full_tracker = pd.json_normalize(tracker, record_path='objects', meta=['frameId']).rename(columns={'id': 'UniqueID'})

new = pd.merge(counter, full_tracker, on=["UniqueID", "frameId"], how="left")
new.drop_duplicates(keep='first')

new[['w', 'h']]

new['x_c'] = new['x'] + new['w']/2
new['y_c'] = new['y'] + new['h']/2

new[['frameId', 'UniqueID', 'ObjectClass', 'bearing_og', 'bearing', 'x_c', 'y_c']]

# IMG STUFF

def save_frame(frame_number, source, arrows=None):
    vc = cv2.VideoCapture(source)
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    rval, frame = vc.read()
    if arrows != None:
        for a in arrows:
            frame = cv2.arrowedLine(frame, a['start'], a['end'], (0,0,255), 5)
    cv2.imwrite(str(frame_number) + '.jpg', frame)

arrows = [{'start': (1430, 441), 'end': (1500, 500)}]

video_location = "../data/short_copy.mp4"
save_frame(181, video_location, arrows=arrows)

cv2.destroyAllWindows()
