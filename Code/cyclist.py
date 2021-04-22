import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import cv2
import sys
import pickle
from sort import *

class Camera:
    def __init__(self, user, video_folder, file_name, camera):
        self.user = user
        self.video_folder = video_folder
        self.file_name = file_name
        self.camera = camera

        self.parent_path = f"/Users/{self.user}/Library/Mobile Documents/com~apple~CloudDocs/Bachelor Project/"
        self.parent_path_video = f"{self.parent_path}Videos/{self.video_folder}/"
        self.tracker_path = f"{self.parent_path_video}Data/{self.file_name}/tracker_{self.file_name}.json"
        self.video_path = f"{self.parent_path_video}Processed/{self.file_name}.mp4"
        self.photo_path = f"{self.parent_path_video}Photos/{self.file_name}"
        self.base_image = f"{self.parent_path}Base Image"
        self.map_path = f"{self.base_image}/FullHD_bridge.png"
        self.temp = []
        self.img = 0

    def df_format(self):
        self.tracker_df["x"] = (self.tracker_df["xmax"] - self.tracker_df["xmin"])/2 + self.tracker_df["xmin"]
        self.tracker_df["y"] = (self.tracker_df["ymax"] - self.tracker_df["ymin"])/2 + self.tracker_df["ymin"]
        self.tracker_df["camera"] = self.camera 
        if "unique_id" not in self.tracker_df.columns:
            self.tracker_df["unique_id"] = None
        self.tracker_df["color"] = None

    def read_pkl(self, name):
        self.file_name = name
        self.tracker_df = pd.read_pickle(f"data/{name}.pickle")
        self.df_format()

    def cyclist_contact_coordiantes(self):
        self.tracker_df["y"] = self.tracker_df["ymax"]

    def smooth_tracks(self, smoothing_factor):
        self.tracker_df["x"] = self.tracker_df.groupby("unique_id")["x"].transform(lambda x: x.rolling(min_periods=1, center=True, window=smoothing_factor).mean())
        self.tracker_df["y"] = self.tracker_df.groupby("unique_id")["y"].transform(lambda y: y.rolling(min_periods=1, center=True, window=smoothing_factor).mean())

    def cut_tracks_with_few_points(self, n):
        self.tracker_df = self.tracker_df[self.tracker_df.groupby("unique_id")["unique_id"].transform("size") > n]

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

    def click_coordinates(self, image, type = "load"):
        if type == "load":
            try:
                with open(f"states/{self.camera}.pickle", "rb") as file:
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
                with open(f"states/{self.camera}.pickle", 'wb') as file:
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
                    color_list.append((0, 0, 255))  # row["color"])
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
        self.tracker_df = self.tracker_df.reset_index()
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
        self.tracker_df_removed = self.tracker_df[self.tracker_df.index.isin(remove_list) == False]

    def new_min_max(self, n):
        self.tracker_df["xmin"] = self.tracker_df["x"] - n
        self.tracker_df["ymin"] = self.tracker_df["y"] - n
        self.tracker_df["xmax"] = self.tracker_df["x"] + n
        self.tracker_df["ymax"] = self.tracker_df["y"] + n

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
            with open(f"data/{name}.pickle", 'wb') as file:
                pickle.dump(self.tracker_df, file)
                print("\n" + "Unique ID's saved")
        elif save_load == "load":
            try:
                name = self.file_name + "_unique_id"
                with open(f"data/{name}.pickle", "rb") as file:
                    self.tracker_df = pickle.load(file)
                print("DF with Unique ID's loaded")
            except FileNotFoundError:
                print("File does not exist")
        else:
            print("Pass 'new' or 'load'")

if __name__ == "__main__":
    g6 = Camera("hogni", 24032021, "2403_g6_sync", "g6")
    g6.read_pkl("2403_g6_sync_yolov5x6")
    # g6.file_name = ""
    g6.unique_id(max_age=90, min_hits=1, iou_threshold=0.10, save_load = "load")
    g6.cyclist_contact_coordiantes()
    g6.get_frame(1000)
    g6.smooth_tracks(20)
    #g6.cut_tracks_with_few_points(50)
    src = g6.click_coordinates(g6.frame, "new")
    src
    dst = g6.click_coordinates(g6.map_path)
    g6.find_homography_matrix(src, dst)
    # warped = g6.warped_perspective(g6.frame, g6.map_path)
    # g6.show_data("Warped img", warped)
    g6.transform_points()
    # plotted_points = g6.plot_object(g6.tracker_df, g6.map_path)
    # g6.show_data("Points", plotted_points)
    remove_line = g6.click_coordinates(g6.map_path)
    g6.remove_point_line(remove_line, "below")
    plotted_removed = g6.plot_object(g6.tracker_df_removed, g6.map_path)
    g6.show_data("Warped img", plotted_removed)

    s7 = Camera("hogni", 24032021, "2403_s7_sync", "s7")
    s7.read_pkl("2403_s7_sync_yolov5x6")
    # s7.file_name = ""
    s7.unique_id(max_age=90, min_hits=1, iou_threshold=0.10, save_load = "load")
    s7.cyclist_contact_coordiantes()
    s7.smooth_tracks(20)
    s7.get_frame(1000)
    src = s7.click_coordinates(s7.frame, "new")
    #s7.cut_tracks_with_few_points(50)
    src = s7.click_coordinates(s7.frame)
    dst = s7.click_coordinates(s7.map_path)
    s7.find_homography_matrix(src, dst)
    #warped = s7.warped_perspective(s7.frame, s7.map_path)
    #s7.show_data("Warped img", warped)
    s7.transform_points()
    #plotted_points = s7.plot_object(s7.tracker_df, s7.map_path)
    #s7.show_data("Points", plotted_points)
    s7.remove_point_line(remove_line, "above")
    plotted_removed_ = s7.plot_object(s7.tracker_df_removed, s7.map_path)
    s7.show_data("Warped img", plotted_removed_)

    g6.tracker_df.to_csv("test.csv")
    g6.tracker_df.to_pickle("g6_processed_not_cut.pickle")

    pd.read_csv("init.csv")