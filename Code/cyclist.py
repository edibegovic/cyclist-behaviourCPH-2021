import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import cv2
import sys
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
        self.tracker_df["unique_id"] = None
        self.tracker_df["color"] = None

    def read_pkl(self, path):
        self.tracker_df = pd.read_pickle(path)
        self.df_format()

    def cyclist_contact_coordiantes(self):
        self.tracker_df["y"] = self.tracker_df["ymax"]

    def smooth_tracks(self, smoothing_factor):
        df_x = (self.tracker_df.groupby("unique_id")["x"].rolling(smoothing_factor, min_periods=1).mean().to_frame(name="smooth_x").droplevel("unique_id"))
        df_y = (self.tracker_df.groupby("unique_id")["y"].rolling(smoothing_factor, min_periods=1).mean().to_frame(name="smooth_y").droplevel("unique_id"))
        self.tracker_df["x"] = df_x
        self.tracker_df["y"] = df_y

    def cut_tracks_with_few_points(self, n):
        self.tracker_df = self.tracker_df[self.tracker_df.groupby("unique_id")["unique_id"].transform("size") > n]

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

    def click_coordinates(self, image):
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

    def unique_id(self, max_age=30, min_hits=0, iou_threshold=0.10):
        tracker = Sort(max_age, min_hits, iou_threshold)
        self.tracker_df = self.tracker_df.sort_values(by="frame_id").reset_index(drop = True)
        frames = sorted(list(set(g6.tracker_df["frame_id"])))

        results = []
        index = []
        for count, i in enumerate(frames):
            temp = []
            temp_index = []
            group = self.tracker_df[self.tracker_df["frame_id"] == i]
            if count % 1000:
                sys.stdout.write("\r" + f"Calculating Unique ID's - {round((count/len(frames))*100, 2)} %")
                sys.stdout.flush()
            for idx, row in group.iterrows():
                temp_index.append(idx)
                temp.append(row[["xmin", "ymin", "xmax", "ymax", "confidence"]])
            results.append(tracker.update(np.array(temp)).tolist())
            index.append(temp_index)

        unique_id_list = []
        for count, i in enumerate(results):
            if count % 1000:
                sys.stdout.write("\r" + f"Writting unique ID's - {round((count/len(results))*100, 2)} %")
                sys.stdout.flush()
            for a in i:
                unique_id_list.append(int(a[-1]))
        self.tracker_df["unique_id"] = unique_id_list


if __name__ == "__main__":
    g6 = Camera("hogni", 24032021, "2403_g6_sync", "g6")
    g6.read_pkl("2403_g6_sync_yolov5x6.pickle")
    g6.unique_id()
    g6.cyclist_contact_coordiantes()
    g6.get_frame(1000)
    src = g6.click_coordinates(g6.frame)
    dst = g6.click_coordinates(g6.map_path)
    g6.find_homography_matrix(src, dst)
    warped = g6.warped_perspective(g6.frame, g6.map_path)
    g6.show_data("Warped img", warped)
    g6.transform_points()
    plotted_points = g6.plot_object(g6.tracker_df, g6.map_path)
    g6.show_data("Points", plotted_points)
    remove_line = g6.click_coordinates(g6.map_path)
    g6.remove_point_line(remove_line, "below")
    plotted_removed = g6.plot_object(g6.tracker_df_removed, g6.map_path)
    g6.show_data("Warped img", plotted_removed)

    g6.tracker_df.to_csv("joined.csv")
    g6.tracker_df.to_pickle("joined.pickle")
  