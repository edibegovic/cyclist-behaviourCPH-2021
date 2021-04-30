import numpy as np
import cv2
import sys
import pickle
from sort import *
import pandas as pd
pd.options.mode.chained_assignment = None
import math
import matplotlib.pyplot as plt
from rdp import rdp
import similaritymeasures as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Camera:
    def __init__(self, user, video_folder, file_name, camera):
        self.user = user
        self.video_folder = video_folder
        self.file_name = file_name
        self.camera = camera

        self.parent_path = f"Data/{self.video_folder}"
        self.video_path = f"{self.parent_path}/Videos/Processed/{self.file_name}.mp4"
        self.map_path = f"{self.parent_path}/Data/Assets/dbro_map.png"
        self.temp = []
        self.img = 0

    def df_format(self):
        self.tracker_df["x"] = (self.tracker_df["xmax"] - self.tracker_df["xmin"])/2 + self.tracker_df["xmin"]
        self.tracker_df["y"] = (self.tracker_df["ymax"] - self.tracker_df["ymin"])/2 + self.tracker_df["ymin"]
        self.tracker_df["camera"] = self.camera 
        if "unique_id" not in self.tracker_df.columns:
            self.tracker_df["unique_id"] = None
        if "color" not in self.tracker_df.columns:
            self.tracker_df["color"] = None
        if "confidence" not in self.tracker_df.columns:
            self.tracker_df["confidence"] = 1

    def read_pkl(self, name):
        self.file_name = name
        self.tracker_df = pd.read_pickle(f"{self.parent_path}/Data/TrackerDF/{name}.pickle")
        self.df_format()

    def cyclist_contact_coordiantes(self):
        self.tracker_df["y"] = self.tracker_df["ymax"]

    def smooth_tracks(self, smoothing_factor):
        self.tracker_df["x"] = self.tracker_df.groupby("unique_id")["x"].transform(lambda x: x.rolling(min_periods=1, center=True, window=smoothing_factor).mean())
        self.tracker_df["y"] = self.tracker_df.groupby("unique_id")["y"].transform(lambda y: y.rolling(min_periods=1, center=True, window=smoothing_factor).mean())

    def smooth_bearings(self, smoothing_factor):
        self.tracker_df["bearing"] = self.tracker_df.groupby("unique_id")["bearing"].transform(lambda x: x.rolling(min_periods=1, center=True, window=smoothing_factor).mean())

    def cut_tracks_with_few_points(self, n):
        self.tracker_df = self.tracker_df[self.tracker_df.groupby("unique_id")["unique_id"].transform("size") > n]

    def add_bearing(self):
        self.tracker_df["bearing"] = 0
        self.tracker_df = self.tracker_df.sort_values(["unique_id", "frame_id"]).reset_index(drop=True)
        len_df = len(self.tracker_df)
        previous_row = []
        unique_id = 0
        for count, (_, row) in enumerate(self.tracker_df.iterrows()):
            if unique_id != row["unique_id"]:
                previous_row = row
            bearing = self.calculate_bearing(row, previous_row)
            self.tracker_df["bearing"][_] = bearing
            if not count % 100:
                    sys.stdout.write("\r" + f"Adding bearing: {round(((count)/(len_df))*100, 3)}%")
                    sys.stdout.flush()
            previous_row = row
            unique_id = row["unique_id"]

    def add_label(self):
        self.tracker_df["label"] = 0
        self.tracker_df = self.tracker_df.sort_values(["unique_id", "frame_id"]).reset_index(drop=True)
        len_df = len(self.tracker_df)
        unique_id = 0
        label = -1
        for count, (_, row) in enumerate(self.tracker_df.iterrows()):
            if unique_id != row["unique_id"]:
                unique_id = row["unique_id"]
                label += 1
            self.tracker_df["label"][_] = self.labels[label]
            if not count % 100:
                    sys.stdout.write("\r" + f"Adding label: {round(((count)/(len_df))*100, 3)}%")
                    sys.stdout.flush()
            unique_id = row["unique_id"]


    def calculate_bearing(self, row, previous_row):
        x_1, y_1 = row["x"], row["y"]
        x_2, y_2 = previous_row["x"], previous_row["y"]
        angle = math.atan2(y_1-y_2, x_1-x_2)
        bearing = math.degrees(angle)
        bearing = (bearing + 360) % 360
        return bearing

    def get_frame(self, frame_number):
        vc = cv2.VideoCapture(self.video_path)
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        rval, self.frame = vc.read()

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp.append([x, y])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(self.img, (x, y), 10, (200, 90, 255), -1)
            cv2.putText(self.img, str(len(self.temp)), (x + 5, y - 5), font, 2, (255, 255, 255), 5)
            cv2.imshow("image", self.img)

    def click_coordinates(self, image, dst = 0, type = "load"):
        if type == "load":
            try:
                with open(f"{self.parent_path}/Data/States/{self.camera}_{dst}.pickle", "rb") as file:
                    self.temp = pickle.load(file)
                print("Coordinates loaded")
            except FileNotFoundError:
                print("File does not exist")
        elif type == "new" or type == "line":
            if self.temp:
                self.temp = []
                self.img = 0

            if isinstance(image, str):
                self.img = cv2.imread(image, 1)
            else:
                self.img = image.copy()

            cv2.imshow("image", self.img)
            cv2.setMouseCallback("image", self.click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            if type != "line":
                with open(f"{self.parent_path}/Data/States/{self.camera}_{dst}.pickle", 'wb') as file:
                    pickle.dump(self.temp, file)
        return self.temp

    def find_homography_matrix(self, source_list, destination_list):
        coordiantes_on_source = np.array(source_list)
        coordiantes_on_destination = np.array(destination_list)
        self.matrix, _ = cv2.findHomography(coordiantes_on_source, coordiantes_on_destination)

    def warped_perspective(self, src, dst):
        if isinstance(src, str):
            source_image = cv2.imread(src)
        else:
            source_image = src.copy()

        if isinstance(dst, str):
            destination_image = cv2.imread(dst)
        else:
            destination_image = dst.copy()
        return cv2.warpPerspective(source_image, self.matrix, (destination_image.shape[1], destination_image.shape[0]))

    def show_data(self, name, cv2_object):
        cv2.imshow(name, cv2_object)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def transform_points(self):
        transformed_x = []
        transformed_y = []

        for _, row in self.tracker_df.iterrows():
            point = (row["x"], row["y"])

            transformed_x.append(int(round((self.matrix[0][0] * point[0] + self.matrix[0][1] * point[1] + self.matrix[0][2]) / ((self.matrix[2][0] * point[0] + self.matrix[2][1] * point[1] + self.matrix[2][2])))))
            transformed_y.append(int(round((self.matrix[1][0] * point[0] + self.matrix[1][1] * point[1] + self.matrix[1][2]) / ((self.matrix[2][0] * point[0] + self.matrix[2][1] * point[1] + self.matrix[2][2])))))

        self.tracker_df["x"] = transformed_x
        self.tracker_df["y"] = transformed_y
    
    def plot_object(self, df, dst_image):
        if isinstance(dst_image, str):
            image = cv2.imread(dst_image)
        else:
            image = dst_image.copy()

        df["x"] = df["x"].astype(int)
        df["y"] = df["y"].astype(int)
        grouped = df.groupby("unique_id")
        if len(grouped) > 1:
            for _, group in grouped:
                xy = []
                color_list = []
                for count, (_, row) in enumerate(group.iterrows()):
                    xy.append((row["x"], row["y"]))
                    color_list.append(row["color"])
                    if len(xy) > 1:
                        image = cv2.line(image, xy[count - 1], xy[count], color_list[count], 3)
        else:
            for _, row in df.iterrows():
                image = cv2.circle(image, (row["x"], row["y"]), 3, (0, 0, 255), -1)
        return image

    def point_inside_polygon(self, x, y, poly):
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def remove_point_polygon(self, poly, remove="inside"):
        remove_list = []
        len_df = len(self.tracker_df)
        for count, (index, row) in enumerate(self.tracker_df.iterrows()):
            x = row["x"]
            y = row["y"]
            if self.point_inside_polygon(x, y, poly):
                if remove == "inside":
                    remove_list.append(index)
            else:
                if remove == "outside":
                    remove_list.append(index)
            if (count % 100) == 0:
                sys.stdout.write("\r" + f"Removal progress - {round((count/len_df)*100, 2)} %")
                sys.stdout.flush()
        self.tracker_df_removed = self.tracker_df[self.tracker_df.index.isin(remove_list) == False]

    def remove_point_line(self, line, remove="above"):
        self.tracker_df = self.tracker_df.reset_index(drop=True)
        remove_list = []
        len_df = len(self.tracker_df)
        for count, (index, row) in enumerate(self.tracker_df.iterrows()):
            x = row["x"]
            y = row["y"]
            above_below = np.cross(np.array([x, y])-np.array(line[0]), np.array(line[1])-np.array(line[0]))
            if above_below > 0:
                if remove == "above":
                    remove_list.append(index)
            elif above_below < 0:
                if remove == "below":
                    remove_list.append(index)    
            if (count % 100) == 0:
                sys.stdout.write("\r" + f"Removal progress - {round((count/len_df)*100, 2)} %")
                sys.stdout.flush()
        self.tracker_df = self.tracker_df[self.tracker_df.index.isin(remove_list) == False]

    def new_bbox(self, bbox_size):
        self.tracker_df["xmin"] = self.tracker_df["x"] - bbox_size
        self.tracker_df["ymin"] = self.tracker_df["y"] - bbox_size
        self.tracker_df["xmax"] = self.tracker_df["x"] + bbox_size
        self.tracker_df["ymax"] = self.tracker_df["y"] + bbox_size

    def unique_id(self, max_age=30, min_hits=1, iou_threshold=0.15, save_load = 0):
        if save_load == "new":
            tracker = Sort(max_age, min_hits, iou_threshold)
            self.tracker_df = self.tracker_df.sort_values(by="frame_id").reset_index(drop = True)
            new_df = pd.DataFrame(columns = ["xmin", "ymin", "xmax", "ymax", "unique_id", "frame_id"])
            max_frame = max(self.tracker_df["frame_id"])

            for i in range(max_frame):
                temp = []
                group = self.tracker_df[self.tracker_df["frame_id"] == i]
                if len(group) != 0:
                    if not i % 10:
                        sys.stdout.write("\r" + f"Calculating Unique ID's - {round((i/max_frame)*100, 2)} %")
                        sys.stdout.flush()
                    for _, row in group.iterrows():
                        temp.append(row[["xmin", "ymin", "xmax", "ymax", "confidence"]])
                    unique_id = tracker.update(np.array(temp))
                    unique_id = [y.tolist()+[i] for y in unique_id]
                    new_df = new_df.append(pd.DataFrame(unique_id, columns=new_df.columns))
            self.tracker_df = new_df
            self.df_format()
            name = self.file_name + "_unique_id"
            with open(f"{self.parent_path}/Data/TrackerDF/{name}.pickle", 'wb') as file:
                pickle.dump(self.tracker_df, file)
                print("\n" + "Unique ID's saved")
        elif save_load == "load":
            try:
                name = self.file_name + "_unique_id"
                with open(f"{self.parent_path}/Data/TrackerDF/{name}.pickle", "rb") as file:
                    self.tracker_df = pickle.load(file)
                print("DF with Unique ID's loaded")
            except FileNotFoundError:
                print("File does not exist")
        else:
            print("Pass 'new' or 'load'")

# Color paths

    def add_color(self, type = "label"):
        self.tracker_df = self.tracker_df.sort_values(["unique_id", "frame_id"]).reset_index(drop=True)
        len_df = len(self.tracker_df)
        colour_list = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
        (255, 255, 255),
        (0, 0, 0)]
        for count, (_, row) in enumerate(self.tracker_df.iterrows()):
            if type == "label":
                color = colour_list[row["label"]]
            else:
                color = self.get_color(int(round(row["bearing"])))
            self.tracker_df["color"][_] = [int(round(color[0]*255)), int(round(color[1]*255)), int(round(color[2]*255))]
            if not count % 100:
                    sys.stdout.write("\r" + f"Adding color Total: {round(((count)/(len_df))*100, 3)}%")
                    sys.stdout.flush()

    def get_color(self, n):
        return plt.cm.gist_ncar(int(round(n)))
        
# Ramer-Douglas-Peucker algorithm - Dimensinality reduction of line segments

    def ramer_reduction(self):
        self.tracker_df = self.tracker_df.sort_values(["unique_id", "frame_id"]).reset_index(drop=True)

        unique_id = 0
        unique_id_list = []
        line_list = []
        len_tracker_df = len(self.tracker_df)
        temp = []

        for count, (_, row) in enumerate(self.tracker_df.iterrows()):
            if count == 0:
                unique_id = row["unique_id"]
                unique_id_list.append(row["unique_id"])
            elif unique_id != row["unique_id"]:
                unique_id_list.append(row["unique_id"])
                unique_id = row["unique_id"]
                line_list.append(temp)
                temp = []
            temp.append([row["x"], row["y"]])

            if count == len_tracker_df - 1:
                line_list.append(temp)
                temp = []

            if not count%100:
                sys.stdout.write("\r" + f"Ramer-Douglas-Peucker reduction step 1: {round(((count)/(len_tracker_df))*100, 3)}%")
                sys.stdout.flush()

        ramer_list = []

        for count, i in enumerate(line_list):
            i = np.rint(np.array(i).reshape((len(i),2)))
            desiredNumberOfPoints = 100
            epsilon = (len(i) / (3 * desiredNumberOfPoints)) * 2
            ramer_list.append(rdp(i, epsilon=epsilon))
            if not count%100:
                sys.stdout.write("\r" + f"Ramer-Douglas-Peucker reduction step 2: {round(((count)/(len_tracker_df))*100, 3)}%")
                sys.stdout.flush()
        self.ramer_list = ramer_list
        self.ramer_id = unique_id_list

    def distance_matrix(self, type = "load"):
        if type == "load":
            self.dist_matrix = np.load(f"{self.parent_path}/Data/States/{self.file_name}_distance_matrix.npy")
        else:
            self.ramer_reduction()
            similarity = lambda track_1, track_2: sm.frechet_dist(track_1, track_2)
            
            len_ramer = len(self.ramer_list)
            dist_matrix = np.zeros((len_ramer, len_ramer))

            for i, group in enumerate(self.ramer_list):
                shorter_group = self.ramer_list[i:]
                len_sg = len(shorter_group)
                for j, group2 in enumerate(shorter_group):
                    distance_ = similarity(group, group2)
                    dist_matrix[i, j] = distance_
                    dist_matrix[j, i] = distance_
                    if not j%10:
                        sys.stdout.write("\r" + f"Distance matrix total: {round(((i)/(len_ramer))*100, 3)}% - current group: {round(((j)/(len_sg))*100, 3)}%")
                        sys.stdout.flush()
            np.save(f"{self.parent_path}/Data/States/{self.file_name}_distance_matrix", dist_matrix)
            self.dist_matrix = dist_matrix

    def calculate_best_n_clusters(self, range_n_clusters=[5,6,7,8,9,10]):
        score, n, labels, model = [], [], [], []
        for n_clusters in range_n_clusters:
            k_model = self.k_mean_model(self.dist_matrix, n_clusters)
            cluster_labels = self.k_predict(k_model, self.dist_matrix)
            labels.append(cluster_labels)

            silhouette_avg = silhouette_score(self.dist_matrix, cluster_labels)
            score.append(silhouette_avg)
            n.append(n_clusters)
            model.append(k_model)
            print(
                f"For n_clusters = {n_clusters} The average silhouette_score is : {round(silhouette_avg, 3)}"
            )

        max_value = max(score)
        max_index = score.index(max_value)
        self.k_model = model[max_index]
        self.best_n = n[max_index]
        self.labels = labels[max_index]

    def k_mean_model(self, dist_matrix, n_clusters):
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        k_mean_model = clusterer.fit(dist_matrix)
        return k_mean_model


    def k_predict(self, model, dist_matrix):
        cluster_labels = model.predict(dist_matrix)
        return cluster_labels


if __name__ == "__main__":
    g6 = Camera("hogni", 24032021, "2403_g6_sync", "g6")
    g6.read_pkl("2403_g6_sync_yolov5x6")
    g6.unique_id(max_age=90, min_hits=1, iou_threshold=0.10, save_load = "load")
    g6.cyclist_contact_coordiantes()
    g6.get_frame(1000)
    g6.smooth_tracks(20)
    g6.cut_tracks_with_few_points(50)
    src = g6.click_coordinates(g6.frame, dst = "src", type = "load")
    dst = g6.click_coordinates(g6.map_path, dst = "dst", type = "load")
    g6.find_homography_matrix(src, dst)
    # warped = g6.warped_perspective(g6.frame, g6.map_path)
    # g6.show_data("Warped img", warped)
    g6.transform_points()
    g6.distance_matrix(type = "new")
    g6.calculate_best_n_clusters()
    g6.add_label()
    # g6.add_bearing()
    # g6.smooth_bearings(50)
    g6.add_color(type="label")
    plotted_points = g6.plot_object(g6.tracker_df, g6.map_path)
    g6.show_data("Points", plotted_points)


    remove_line = g6.click_coordinates(g6.map_path, dst = 0, type = "line")
    g6.remove_point_line(remove_line, "below")
    plotted_removed = g6.plot_object(g6.tracker_df, g6.map_path)
    g6.show_data("Warped img", plotted_removed)

    s7 = Camera("hogni", 24032021, "2403_s7_sync", "s7")
    s7.read_pkl("2403_s7_sync_yolov5x6")
    # s7.file_name = ""
    s7.unique_id(max_age=90, min_hits=1, iou_threshold=0.10, save_load = "load")
    s7.cyclist_contact_coordiantes()
    s7.get_frame(1000)
    s7.smooth_tracks(20)
    s7.cut_tracks_with_few_points(10)
    src = s7.click_coordinates(s7.frame, dst = "src", type = "load")
    dst = s7.click_coordinates(s7.map_path, dst = "dst", type = "load")
    s7.find_homography_matrix(src, dst)
    # warped = s7.warped_perspective(s7.frame, s7.map_path)
    # s7.show_data("Warped img", warped)
    s7.transform_points()
    #plotted_points = s7.plot_object(s7.tracker_df, s7.map_path)
    #s7.show_data("Points", plotted_points)
    s7.remove_point_line(remove_line, "above")
    plotted_removed_s7 = s7.plot_object(s7.tracker_df, s7.map_path)
    s7.show_data("Warped img", plotted_removed_s7)

    def join_df(df_list):
        return pd.concat(df_list, ignore_index=True).sort_values("frame_id").reset_index(drop=True)

    joined_df = join_df([g6.tracker_df, s7.tracker_df])
    joined = Camera("hogni", 24032021, "joined", "joined")
    joined.tracker_df = joined_df
    joined.new_bbox(10)
    joined.df_format()
    joined.unique_id(max_age=90, min_hits=1, iou_threshold=0.15, save_load = "load")

    joined.tracker_df.to_csv("Data/24032021/Data/CSV/joined_df_90_1_0.15_bbox10.csv")
    joined.tracker_df

    joined = Camera("hogni", 24032021, "joined", "joined")
    joined.unique_id(max_age=90, min_hits=1, iou_threshold=0.15, save_load = "load")
    # joined.tracker_df["x"] = joined.tracker_df["x"].round(0).astype(int)
    # joined.tracker_df["y"] = joined.tracker_df["y"].round(0).astype(int)