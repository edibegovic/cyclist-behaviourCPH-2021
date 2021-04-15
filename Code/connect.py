import pandas as pd
import math
import project
import numpy as np
import matplotlib.pyplot as plt
import similaritymeasures
import sys
from multiprocessing import Process

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

def pixel_distance(cut_group):
    x_1, y_1 = cut_group.iloc[0]["altered_x"], cut_group.iloc[0]["altered_y"]
    x_2, y_2 = cut_group.iloc[1]["altered_x"], cut_group.iloc[1]["altered_y"]
    distance = math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
    return distance

def relative_velocity(tracker_df):
    pass

def remove_far_field_points(tracker_df, polygon_points):
    polygon = Polygon(polygon_points)
    tracker_df["Inside_polygon"] = False
    for _, row in tracker_df.itterows():
        point = Point(row["altered_x"], row["altered_y"])
        if polygon.contains(point):
            tracker_df.at[_, "Inside_polygon"] = True
    return tracker_df

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
                try:
                    t4 = Process(target=calculate_assign, args=(i, new_index, grouped_df_list, grouped_id_list, 3))
                    t4.start()
                    process_list.append(t4)
                except KeyError:
                    pass

                for t in process_list:
                    t.join()

            except RuntimeError:
                print ("Error: unable to start thread")
            
                if new_index+1 == len_grouped_df:
                    break

        count += 4
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

    tracker_img = morph.get_cv2_point_plot(tracker_df_, (f"{raw_combined.base_image}/{raw_combined.birds_eye_view_image}"), colours = 1) #, joined.labels, joined.uniqueid)
    morph.show_data(tracker_img)