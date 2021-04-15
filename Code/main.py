import trackerdf as tdf
import morph
import ml
import pandas as pd
import time
import pickle
import _thread
import sys


class Cameras:
    def __init__(
        self,
        who_is_running_this_code,
        video_folder,
        file_name,
        frame_number=1000,
        smooth_factor=100,
        cut_df=100,
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

    def run(self):
        self.tracker_df = tdf.create_tracker_df(self.tracker_path)
        self.tracker_df = morph.cyclist_contact_coordiantes(self.tracker_df)
        self.tracker_df = morph.smooth_tracks(self.tracker_df, self.smooth_factor)
        self.tracker_df = morph.cut_tracks_with_few_points(self.tracker_df, self.cut_df)
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

    importlib.reload(ml)

    g6 = Cameras("hogni", 24032021, "2403_G6_sync")
    g6 = g6.load_pickle("g6")
    # g6.run()
    # g6.save_object("g6")
    g6.run_clustering()
    g6.plot_and_show()

    s7 = Cameras("hogni", 24032021, "2403_S7_sync")
    s7 = s7.load_pickle("s7")
    # s7.run()
    # s7.save_object("s7")
    s7.run_clustering()
    s7.plot_and_show()

    iph12 = Cameras("hogni", 24032021, "2403_edi_sync")
    iph12 = iph12.load_pickle("iph12")
    # iph12.run()
    # iph12.save_object("iph12")
    iph12.run_clustering()
    iph12.plot_and_show()

    # joined_df = morph.join_df(g6.tracker_df, s7.tracker_df, iph12.tracker_df)
    joined = Cameras("hogni", 24032021, "2403_G6_sync")
    joined = joined.load_pickle("joined")
    # joined.tracker_df = joined_df
    # joined.save_object("joined")
    _thread.start_new_thread(joined.run_clustering())
    # joined.run_clustering()
    # tracker_img = morph.get_cv2_point_plot(joined.tracker_df, (f"{joined.base_image}/{joined.birds_eye_view_image}"), joined.labels, joined.uniqueid)
    # morph.show_data(tracker_img)

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

    outer = pd.concat([g6.tracker_df, s7.tracker_df])
    outer = pd.concat([outer, iph12.tracker_df])

    outer = outer.sort_values(by=["frameId"])
    outer_grouped = outer.groupby("UniqueID")
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

    import cv2

    def video(outer):
        out = cv2.VideoWriter("output.avi", -1, 10.0, (1920, 1080))

        previous_i = 10000
        len_df = len(outer)
        for i in range(0, len_df, 10000):
            sys.stdout.write("\r" + f"Video Progress {round(((i)/(len_df))*100, 2)}%")
            sys.stdout.flush()
            mor = morph.get_cv2_point_plot(
                outer[i:previous_i], (f"{s7.base_image}/{s7.birds_eye_view_image}")
            )
            out.write(mor)
            previous_i += i

        out.release()
        cv2.destroyAllWindows()

    video(outer)
