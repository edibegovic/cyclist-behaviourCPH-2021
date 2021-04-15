import trackerdf as tdf
import morph
import ml
import pandas as pd
import time
import pickle
import _thread
import sys
from statistics import mean


class Cameras:
    def __init__(
        self,
        who_is_running_this_code,
        video_folder,
        file_name,
        frame_number=1000,
        smooth_factor=100,
        cut_df=20,
        library=0,
        parent_path=0,
        parent_path_video=0,
        tracker_path=0,
        video_path=0,
        photo_path=0,
        base_image=0,
        birds_eye_view_image=0,
    ):
        self.who_is_running_this_code = who_is_running_this_code
        self.video_folder = video_folder
        self.file_name = file_name
        self.frame_number = frame_number
        self.smooth_factor = smooth_factor
        self.cut_df = cut_df

        self.library = "Library"
        self.parent_path = f"/Users/{self.who_is_running_this_code}/{self.library}/Mobile Documents/com~apple~CloudDocs/Bachelor Project/"
        self.parent_path_video = f"{self.parent_path}Videos/{self.video_folder}/"
        self.tracker_path = f"{self.parent_path_video}Data/{self.file_name}/tracker_{self.file_name}.json"
        self.video_path = f"{self.parent_path_video}Processed/{self.file_name}.mp4"
        self.photo_path = f"{self.parent_path_video}Photos/{self.file_name}"
        self.base_image = f"{self.parent_path}Base Image"
        self.birds_eye_view_image = "FullHD_bridge.png"

    def make_raw_df(self, name):
        self.tracker_df = tdf.create_tracker_df(self.tracker_path)
        self.tracker_df = morph.add_camera(self.tracker_df, name)
        self.tracker_df = morph.cyclist_contact_coordiantes(self.tracker_df)

    def smooth(self):
        self.tracker_df = morph.smooth_tracks(self.tracker_df, self.smooth_factor)

    def cut(self):
        self.tracker_df = morph.cut_tracks_with_few_points(self.tracker_df, self.cut_df)
        
    def warp(self):
        morph.capture_image_from_video(
            self.video_path, self.base_image, self.file_name, self.frame_number
        )
        src_image_points = morph.click_coordinates(
            f"{self.base_image}/{self.file_name}.jpg"
        )
        dst_image_points = morph.click_coordinates(
            f"{self.base_image}/{self.birds_eye_view_image}"
        )
        self.homo, status = morph.find_homography_matrix(
            src_image_points, dst_image_points
        )
        self.warped_img = morph.warped_perspective(
            (f"{self.base_image}/{self.file_name}.jpg"),
            (f"{self.base_image}/{self.birds_eye_view_image}"),
            self.homo,
        )
        morph.show_data(self.warped_img)
        self.tracker_df = morph.transform_points(self.tracker_df, self.homo)
        self.tracker_df = morph.smooth_tracks(self.tracker_df, self.smooth_factor)

    def run_clustering(self):
        (
            self.n_clusters,
            self.labels,
            self.uniqueid,
            self.model,
            self.vectors,
        ) = ml.run_all(self.tracker_df)

    def plot_and_show(self):
        try:
            self.tracker_img = morph.get_cv2_point_plot(
                self.tracker_df,
                (f"{self.base_image}/{self.birds_eye_view_image}"),
                self.labels,
                self.uniqueid,
            )
        except AttributeError:
            self.tracker_img = morph.get_cv2_point_plot(
                self.tracker_df, (f"{self.base_image}/{self.birds_eye_view_image}")
            )
        morph.show_data(self.tracker_img)

    def save_object(self, name):
        with open(f"{name}.pickle", "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, name):
        with open(f"{name}.pickle", "rb") as input:
            return pickle.load(input)

if __name__ == "__main__":

    # Reload a module.
    import importlib
    importlib.reload(morph)

    g6 = Cameras("hogni", 24032021, "2403_G6_sync")
    # g6 = g6.load_pickle("g6_smooth_5_all_points")
    g6.make_raw_df("g6")
    g6.warp()
    g6.save_object("g6_raw")
    # g6.run_clustering()
    # g6.plot_and_show()

    s7 = Cameras("hogni", 24032021, "2403_S7_sync")
    # s7 = s7.load_pickle("s7_smooth_5_all_points")
    s7.make_raw_df("s7")
    s7.warp()
    s7.save_object("s7_raw")
    # s7.run_clustering()
    # s7.plot_and_show()

    iph12 = Cameras("hogni", 24032021, "2403_edi_sync")
    # iph12 = iph12.load_pickle("iph12_smooth_5_all_points")
    iph12.make_raw_df("iph12")
    iph12.warp()
    iph12.save_object("iph12_raw")
    # iph12.run_clustering()
    # iph12.plot_and_show()

    raw_combined = Cameras("hogni", 24032021, "2403_G6_sync")
    raw_combined_df = morph.join_df(g6.tracker_df, s7.tracker_df, iph12.tracker_df)
    raw_combined_df
    raw_combined = raw_combined.load_pickle("raw_combined")
    raw_combined.tracker_df = raw_combined_df

    raw_combined_df.to_csv("raw_combined_df")

    # joined.tracker_df = joined_df
    raw_combined.save_object("raw_combined")
    # _thread.start_new_thread(joined.run_clustering())
    # joined.run_clustering()
    tracker_img = morph.get_cv2_point_plot(joined.tracker_df, (f"{joined.base_image}/{joined.birds_eye_view_image}"), colours = 1) #, joined.labels, joined.uniqueid)
    morph.show_data(tracker_img)
    joined.tracker_df.sort_values(by=["frameId"])[100:150]
    joined.tracker_df

    tracker_img = morph.get_cv2_point_plot(joined.tracker_df[1000:2000], (f"{joined.base_image}/{joined.birds_eye_view_image}")) #, joined.labels, joined.uniqueid)
    morph.show_data(tracker_img)

    tracker_img = morph.get_cv2_point_plot(joined.tracker_df[4000:5000], (f"{joined.base_image}/{joined.birds_eye_view_image}")) #, joined.labels, joined.uniqueid)
    morph.show_data(tracker_img)

    tracker_img = morph.get_cv2_point_plot(joined.tracker_df[5000:6000], (f"{joined.base_image}/{joined.birds_eye_view_image}")) #, joined.labels, joined.uniqueid)
    morph.show_data(tracker_img)

    tracker_img = morph.get_cv2_point_plot(joined.tracker_df[6000:7000], (f"{joined.base_image}/{joined.birds_eye_view_image}")) #, joined.labels, joined.uniqueid)
    morph.show_data(tracker_img)

    joined.tracker_df[1000:2000]

    tracker_img = morph.get_cv2_point_plot(joined.tracker_df, (f"{joined.base_image}/{joined.birds_eye_view_image}")) #, joined.labels, joined.uniqueid)
    morph.show_data(tracker_img)

    # count = 0
    # for i in range(max(g6.tracker_df["frameId"]):
    #     g6_track = g6.tracker_df[g6.tracker_df["frameId"] == i]
    #     uni = morph.get_cv2_point_plot(g6_track, (f"{g6.base_image}/{g6.birds_eye_view_image}"))
    #     morph.show_data(uni)

    # count = 0
    # for name, group in s7.tracker_df.groupby("UniqueID"):
    #     s7_track = s7.tracker_df[s7.tracker_df["UniqueID"] == name]
    #     uni_s7 = morph.get_cv2_point_plot(s7_track, (f"{s7.base_image}/{s7.birds_eye_view_image}"))
    #     morph.show_data(uni_s7)
    #     count += 1
    #     if count == 5:
    #         break

    # outer = pd.concat([g6.tracker_df, s7.tracker_df])
    # outer = pd.concat([outer, iph12.tracker_df])

    # outer = outer.sort_values(by=["frameId"])
    # outer = Cameras("hogni", 24032021, "2403_edi_sync")
    # x.tracker_df = outer
    # x.save_object("concatenated")
    # outer = outer.load_pickle("concatenated")
    # outer_grouped = outer.tracker_df.groupby("UniqueID")
    # outer.tracker_df[1000:1100]
    # frame_id = set()
    # un_id = []
    # for count, (name, row) in enumerate(outer.iterrows()):
    #     un_id.append(row["UniqueID"])
    #     frame_id.add(row["frameId"])
    #     if len(frame_id) == 100:
    #         df = outer[outer["UniqueID"].isin(un_id)]
    #         mor = morph.get_cv2_point_plot(df, (f"{s7.base_image}/{s7.birds_eye_view_image}"))
    #         morph.show_data(mor)
    #         time.sleep(1)
    #         frame_id = set()
    #     if count == 20000:
    #         break

    # import cv2

    # def video(outer):
    #     # out = cv2.VideoWriter("output.avi", -1, 10.0, (1920, 1080))

    #     previous_i = 1000
    #     len_df = len(outer)
    #     for i in range(0, len_df, 1000):
    #         # sys.stdout.write("\r" + f"Video Progress {round(((i)/(len_df))*100, 2)}%")
    #         # sys.stdout.flush()
    #         mor = morph.get_cv2_point_plot(
    #             outer[i:previous_i], (f"{s7.base_image}/{s7.birds_eye_view_image}")
    #         )
    #         morph.show_data(mor)
    #         # out.write(mor)
    #         previous_i += i

    #     # out.release()
    #     # cv2.destroyAllWindows()

    # video(outer)

    ########
    # Trying to join
    # outer_grouped = outer.tracker_df.groupby("UniqueID")
    # previous_i = 30
    # len_df = len(outer.tracker_df)
    # len_df**3

    # def are_overlapping(point_one, point_two):
    #     return not(point_one[1] < point_two[0] or point_two[1] < point_one[0])

    # for i in range(0, len_df, 30):
    #     unique_id = outer.tracker_df[i:previous_i].UniqueID.unique()
    #     for id in unique_id:
    #         track_one = outer.tracker_df[outer.tracker_df["UniqueID"] == id]

    #         for a in range(1, len(unique_id)):
    #             track_two = outer.tracker_df[outer.tracker_df["UniqueID"] == unique_id[a]]

    #             # Check bearings aboutt eh same
    #             abs_1 = mean(track_one["bearing"])
    #             abs_2 = mean(track_two["bearing"])
    #             if abs_1 > abs_2:
    #                 check = abs_1 - abs_2
    #             else:
    #                 check = abs_2 - abs_1
    #             if check <= 90:
    #                 over = []
    #                 # print(track_one)
    #                 for _, b in track_one.iterrows():
    #                     for _, c in track_two.iterrows():
    #                         overlapping = are_overlapping((b["x"],b["y"]), (c["x"],c["y"]))
    #                         if overlapping:
    #                             over.append(1)
    #                 if len(over) >= 10:
    #                     outer.tracker_df["UniqueID"].loc[(outer.tracker_df["UniqueID"] == unique_id[a])] = id
    #                 over = []
    #     previous_i += i

    # outer = Cameras("hogni", 24032021, "2403_edi_sync")
    # outer = outer.load_pickle("concatenated")
    # outer.tracker_df

    # new_df =pd.DataFrame(columns=["UniqueID", "frameId", "bearing", "altered_x", "altered_y", "camera"])
    # for name, group in outer.tracker_df.groupby("frameId"):
    #     for _, i in group.iterrows():
    #         pass