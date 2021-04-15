from math import atan2, cos, sin, degrees
import pandas as pd
import math
import morph
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
import numpy as np
import matplotlib.pyplot as plt
# from frechetdist import frdist
import similaritymeasures
import sys
from multiprocessing import Process
# import threading
pd.options.mode.chained_assignment = None

def itterate_through_df_in_chunks_and_apply(tracker_df):
    tracker_df_grouped = tracker_df.groupby("UniqueID")
    for name, group in tracker_df_grouped:
        length = len(group)
        last_bearing = dict()
        last_distance = dict()

        for i in range(length):
            previous_i = i + 2
            cut_group = group[i:previous_i].sort_values(by=["frameId"])
            len_small_group = len(cut_group)
            if len_small_group == 1:
                try:
                    tracker_df.at[cut_group.index[0], "new_bearing"] = last_bearing[cut_group.iloc[0]["UniqueID"]]
                    # tracker_df.at[cut_group.index[0], "Distance"] = last_distance[cut_group.iloc[0]["UniqueID"]]
                except KeyError:
                    tracker_df.at[cut_group.index[0], "new_bearing"] = 0
                    # tracker_df.at[cut_group.index[0], "Distance"] = 0
            else:
                bearing = calculate_bearing(cut_group)
                # distance = pixel_distance(cut_group)
                tracker_df.at[cut_group.index[0], "new_bearing"] = bearing
                # tracker_df.at[cut_group.index[0], "Distance"] = distance
                last_bearing[cut_group.iloc[0]["UniqueID"]] = bearing
                # last_distance[cut_group.iloc[0]["UniqueID"]] = distance
                
                if length == 2:
                    tracker_df.at[cut_group.index[1], "new_bearing"] = bearing
                    # tracker_df.at[cut_group.index[1], "Distance"] = distance
    return tracker_df

def calculate_bearing(cut_group):
    x_1, y_1 = 960, 540
    # x_1, y_1 = cut_group.iloc[0]["smooth_x"], cut_group.iloc[0]["smooth_y"]
    x_2, y_2 = cut_group.iloc[1]["smooth_x"], cut_group.iloc[1]["smooth_y"]
    # angle = atan2(cos(y_1)*sin(y_2)-sin(y_1) * cos(y_2)*cos(x_2-x_1), sin(x_2-x_1)*cos(y_2))
    # bearing = (degrees(angle) + 360) % 360
    angle = math.atan2(y_1-y_2, x_1-x_2)
    bearing = math.degrees(angle)
    bearing = (bearing + 360) % 360
    return bearing

def add_colour(tracker_df):
    for _, row in tracker_df.iterrows():
        # bearing = calculate_bearing((row["altered_x"], row["altered_y"]))
        colour = get_color(row["new_bearing_smooth"])
        # tracker_df.at[tracker_df.index[_], "new_bearing"] = bearing
        tracker_df.at[tracker_df.index[_], "colour_1"] = colour[0]
        tracker_df.at[tracker_df.index[_], "colour_2"] = colour[1]
        tracker_df.at[tracker_df.index[_], "colour_3"] = colour[2]
        tracker_df.at[tracker_df.index[_], "colour_4"] = colour[3]
    return tracker_df

def get_color(n):
    return plt.cm.gist_ncar(int(round(n)))

def smooth_bearings(tracker_df):
    tracker_df_ = (
        tracker_df.groupby("UniqueID")["new_bearing"]
        .rolling(20, min_periods=1)
        .mean()
        .to_frame(name="new_bearing_smooth")
        .droplevel("UniqueID")
    )
    tracker_df = tracker_df.join(tracker_df_)
    return tracker_df

def run_connect(tracker_df):
    tracker_df = itterate_through_df_in_chunks_and_apply(tracker_df)
    tracker_df = smooth_bearings(tracker_df)
    tracker_df = add_colour(tracker_df)
    return tracker_df

# def calculate_bearing(point):
#     x_1, y_1 = 960, 0
#     x_2, y_2 = point[0], point[1]
#     angle = math.atan2(y_1-y_2, x_1-x_2)
#     bearing = math.degrees(angle)
#     bearing = (bearing + 360) % 360
#     return bearing



# def calculate_bearing(tracker_df):
#     tracker_df_grouped = tracker_df.groupby("UniqueID")
#     for name, group in tracker_df_grouped:
#         length = len(group)
#         last_bearing = dict()

#         for i in range(length):
#             previous_i = i + 2
#             cut_group = group[i:previous_i]
#             len_small_group = len(cut_group)
#             if len_small_group == 1:
#                 try:
#                     tracker_df.at[cut_group.index[0], "new_bearing"] = last_bearing[cut_group.iloc[0]["UniqueID"]]
#                 except KeyError:
#                     tracker_df.at[cut_group.index[0], "new_bearing"] = 0
#             else:
#                 x_1, y_1 = cut_group.iloc[0]["altered_x"], cut_group.iloc[0]["altered_y"]
#                 x_2, y_2 = cut_group.iloc[1]["altered_x"], cut_group.iloc[1]["altered_y"]
#                 angle = atan2(cos(y_1)*sin(y_2)-sin(y_1) * cos(y_2)*cos(x_2-x_1), sin(x_2-x_1)*cos(y_2))
#                 bearing = (degrees(angle) + 360) % 360
#                 tracker_df.at[cut_group.index[0], "new_bearing"] = bearing
#                 last_bearing[cut_group.iloc[0]["UniqueID"]] = bearing
#                 if length == 2:
#                     tracker_df.at[cut_group.index[1], "new_bearing"] = bearing
#     return tracker_df

def pixel_distance(cut_group):
    x_1, y_1 = cut_group.iloc[0]["altered_x"], cut_group.iloc[0]["altered_y"]
    x_2, y_2 = cut_group.iloc[1]["altered_x"], cut_group.iloc[1]["altered_y"]
    distance = math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
    return distance

def relative_velocity(tracker_df):
    pass

# def remove_far_field_points(tracker_df, polygon_points):
#     polygon = Polygon(polygon_points)
#     tracker_df["Inside_polygon"] = False
#     for _, row in tracker_df.itterows():
#         point = Point(row["altered_x"], row["altered_y"])
#         if polygon.contains(point):
#             tracker_df.at[_, "Inside_polygon"] = True
#     return tracker_df

def calculate_assign(i, new_index, grouped_df_list, grouped_id_list, num = 0):
    sim = similaritymeasures.frechet_dist(grouped_df_list[i + num], grouped_df_list[new_index + num])
    fr_matrix.at[grouped_id_list[i + num], grouped_id_list[new_index + num]] = sim
    fr_matrix.at[grouped_id_list[new_index + num], grouped_id_list[i + num]] = sim

def fr_distance_matrix(tracker_df):
    tracker_df["xy_list"] = tracker_df[["smooth_x", "smooth_y"]].values.tolist()

    grouped_df = tracker_df.groupby("UniqueID")
    grouped_df_list = []
    grouped_id_list = []
    len_grouped_df = len(grouped_df)
    count = 0
    process_list = []

    for name, group in grouped_df:
        sys.stdout.write("\r" + f"Constructing lists {round(((count)/(len_grouped_df))*100, 2)}%")
        sys.stdout.flush()
        grouped_id_list.append(name)
        grouped_df_list.append(group["xy_list"].tolist())
        count += 1
    names = [x for x in grouped_id_list]
    fr_matrix = pd.DataFrame(columns=names, index=names)
    count = 0

    for i in range(len_grouped_df):
        sys.stdout.write("\r" + f"FR Distance Matrix progress {round(((count)/(len_grouped_df))*100, 2)}%")
        sys.stdout.flush()
        for idx in range(0, len(grouped_df_list[i:]), 4):
            new_index = i + idx
            try:
                t1 = Process(target=calculate_assign, args=(i, new_index, grouped_df_list, grouped_id_list, 0))
                t1.start()
                process_list.append(t1)
                t2 = Process(target=calculate_assign, args=(i, new_index, grouped_df_list, grouped_id_list, 1))
                t2.start()
                process_list.append(t2)
                t3 = Process(target=calculate_assign, args=(i, new_index, grouped_df_list, grouped_id_list, 2))
                t3.start()
                process_list.append(t3)
                # sim = similaritymeasures.frechet_dist(grouped_df_list[i], grouped_df_list[new_index])
                # fr_matrix.at[grouped_id_list[i], grouped_id_list[new_index]] = sim
                # fr_matrix.at[grouped_id_list[new_index], grouped_id_list[i]] = sim

                # sim = similaritymeasures.frechet_dist(grouped_df_list[i + 1], grouped_df_list[new_index + 1])
                # fr_matrix.at[grouped_id_list[i + 1], grouped_id_list[new_index + 1]] = sim
                # fr_matrix.at[grouped_id_list[new_index + 1], grouped_id_list[i + 1]] = sim

                # sim = similaritymeasures.frechet_dist(grouped_df_list[i + 2], grouped_df_list[new_index + 2])
                # fr_matrix.at[grouped_id_list[i + 2], grouped_id_list[new_index + 2]] = sim
                # fr_matrix.at[grouped_id_list[new_index + 2], grouped_id_list[i + 2]] = sim
                try:
                    t4 = Process(target=calculate_assign, args=(i, new_index, grouped_df_list, grouped_id_list, 3))
                    t4.start()
                    process_list.append(t4)
                    # sim = similaritymeasures.frechet_dist(grouped_df_list[i + 3], grouped_df_list[new_index + 3])
                    # fr_matrix.at[grouped_id_list[i + 3], grouped_id_list[new_index + 3]] = sim
                    # fr_matrix.at[grouped_id_list[new_index + 3], grouped_id_list[i + 3]] = sim
                except KeyError:
                    pass

                for t in process_list:
                    t.join()

            except RuntimeError:
                print ("Error: unable to start thread")
            
                if new_index+1 == len_grouped_df:
                    break

        count += 4
    # for idx, (_, group) in enumerate(grouped_df):
    #     count += 1
    #     group_list = group["xy_list"].tolist()
    #     sys.stdout.write("\r" + f"FR Distance Matrix progress {round(((count)/(len_grouped_df))*100, 2)}%")
    #     sys.stdout.flush()
    #     for idx_2, (_, group_2) in enumerate(grouped_df):
    #         if idx_2 >= idx:
    #             group_2_list = group_2["xy_list"].tolist()
    #             # len_group = len(group_list)
    #             # len_group_2 = len(group_2_list) 
    #             # if len_group > len_group_2:
    #             #     print("G1 " + str(len(group_list)))
    #             #     print("G2 " + str(len(group_2_list)))
    #             #     group_2_list += ([] * (len_group - len_group_2))
    #             #     print("G1_ " + str(len(group_list)))
    #             #     print("G2_ " + str(len(group_2_list)))
    #             # elif len_group < len_group_2:
    #             #     group_list += [0, 0] * (len_group_2 - len_group)
    #             sim = similaritymeasures.frechet_dist(group_list, group_2_list)
    #             fr_matrix.at[idx, idx_2] = sim
    #             fr_matrix.at[idx_2, idx] = sim
    return tracker_df, fr_matrix

def distance_matrix(vectors):
    euclidean_d = lambda vector_u, vector_v: max(
        distance.euclidean(vector_u, vector_v), distance.euclidean(vector_v, vector_u)
    )
    count_unique_id = len(vectors)
    dist_matrix = np.zeros((count_unique_id, count_unique_id))

    count = 0
    count_unique_id_square = count_unique_id ** 2
    for i in range(count_unique_id):
        count += 1
        # print(f"Distance Matrix progress {round(((count)/(count_unique_id_square))*100, 3)}%", end="\r")
        # sys.stdout.write("\r" + f"Distance Matrix progress {round(((count)/(count_unique_id_square))*100, 3)}%")
        # sys.stdout.flush()
        for j in range(i + 1, count_unique_id, 4):
            distance_ = euclidean_d(vectors[i], vectors[j])
            dist_matrix[i, j] = distance_
            dist_matrix[j, i] = distance_
    return dist_matrix

if __name__ == "__main__":

    df, matrix = fr_distance_matrix(tracker_df_1)
    raw_combined.tracker_df
    raw_combined.tracker_df = morph.smooth_tracks(raw_combined.tracker_df, 50)
    tracker_df_ = run_connect(raw_combined.tracker_df)

    raw_combined.tracker_df
    tracker_df_1 = morph.cut_tracks_with_few_points(raw_combined.tracker_df, 1000)
    tracker_df_1
    # x = itterate_through_df_in_chunks_and_apply(joined.tracker_df)
    # x
    tracker_img = morph.get_cv2_point_plot(tracker_df_, (f"{raw_combined.base_image}/{raw_combined.birds_eye_view_image}"), colours = 1) #, joined.labels, joined.uniqueid)
    morph.show_data(tracker_img)

    # joined.tracker_df["colour"] = []
    # type(joined.tracker_df)
    # joined.tracker_df.columns
    # x.sort_values(by=["UniqueID", "frameId"])

    # # remove_far_field_points(g6.tracker_df, [(0, 0), (1000, 800), (800, 700)])

    # plt.hist([1,2],color=uniqueish_color(125))
    # uniqueish_color(3)