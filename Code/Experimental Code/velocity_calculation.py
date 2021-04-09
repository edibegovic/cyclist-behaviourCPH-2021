import pandas as pd
import datetime
import numpy as np
import math

# Velocity Calculations
# ------------------------------------------


def calculate_speed(distance, df):
    speed_dictionary = {}
    for index, row in df.iterrows():
        if row[4] in speed_dictionary.keys():
            speed_dictionary[row[4]].append(row[1])
        else:
            speed_dictionary[row[4]] = [row[1]]

    for key, value in speed_dictionary.copy().items():
        if value[0] > value[1]:
            difference = value[0] - value[1]
        else:
            difference = value[1] - value[0]
        speed_dictionary[key] = difference

    for key, value in speed_dictionary.copy().items():
        t = speed_dictionary[key].total_seconds()
        if t == 0:
            t = 1
            speed_dictionary[key] = distance / t
        else:
            speed_dictionary[key] = distance / t
    return speed_dictionary


# Absolute Heading
# ------------------------------------------


def calculate_true_heading(frame_orientation, bearing):
    true_heading = bearing + frame_orientation
    if true_heading > 360:
        true_heading = true_heading - 360
    return true_heading
