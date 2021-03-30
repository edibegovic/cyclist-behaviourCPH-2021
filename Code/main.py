
import trackerdf as tdf
import morph
import trajectory
import cv2
import matplotlib.pyplot as plt


class Cameras:
    def __init__(
        self,
        who_is_running_this_code,
        video_folder,
        file_name,
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

        self.library = "Library"
        self.parent_path = f"/Users/{self.who_is_running_this_code}/{self.library}/Mobile Documents/com~apple~CloudDocs/Bachelor Project/"
        self.parent_path_video = f"{self.parent_path}Videos/{self.video_folder}/"
        self.tracker_path = f"{self.parent_path_video}Data/{self.file_name}/tracker_{self.file_name}.json"
        self.video_path = f"{self.parent_path_video}Processed/{self.file_name}.mp4"
        self.photo_path = f"{self.parent_path_video}Photos/{self.file_name}"
        self.base_image = f"{self.parent_path}Base Image"
        self.birds_eye_view_image = "FullHD_bridge.png"

    # Make tracker df.
    # ------------------------------------------

    def make_tracker_df(self, smooth_factor=20):
        self.tracker_df = tdf.create_tracker_df(self.tracker_path)
        self.tracker_df = morph.cyclist_contact_coordiantes(self.tracker_df)
        self.tracker_df = morph.smooth_tracks(self.tracker_df, smooth_factor)

    # Cut df
    # ------------------------------------------

    def cut_df(self, number_to_cut=100):
        self.tracker_df = morph.cut_tracks_with_few_points(self.tracker_df, 100)

    # Capture frame from video
    # ------------------------------------------

    def get_frame(self, frame_number=1000):
        self.src_image = morph.capture_image_from_video(
            self.video_path, self.base_image, self.file_name, frame_number
        )

    # Get points on src and dst images
    # ------------------------------------------

    def get_coordinates(self):
        self.src_image_points = morph.click_coordinates(
            f"{self.base_image}/{self.file_name}.jpg"
        )
        self.dst_image_points = morph.click_coordinates(
            f"{self.base_image}/{self.birds_eye_view_image}"
        )

    # Get homography matrix
    # ------------------------------------------

    def homo_matrix(self):
        self.homo, status = morph.find_homography_matrix(
            self.src_image_points, self.dst_image_points
        )

    # Display warped image
    # ------------------------------------------

    def warp_image(self):
        warped_img = morph.warped_perspective(
            (f"{self.base_image}/{self.file_name}.jpg"),
            (f"{self.base_image}/{self.birds_eye_view_image}"),
            self.homo,
        )
        morph.show_data(warped_img)

    # Transformed tracks
    # ------------------------------------------

    def get_transformmed_tracks(self):
        x_list = []
        y_list = []
        for index, row in self.tracker_df.iterrows():
            x_list.append(row["mean_x"])
            y_list.append(row["mean_y"])

        self.transformed_x, self.transformed_y = morph.transform_tracker_data(x_list, y_list, self.homo)

    # Plot tracks
    # ------------------------------------------

    def plot_tracks(self):
        img = morph.show_transformed_tracker_data(self.transformed_x, self.transformed_y, self.base_image, self.birds_eye_view_image)
        morph.show_data(img)

if __name__ == "__main__":

    def run(class_object):
        class_object.make_tracker_df()
        class_object.cut_df()
        class_object.get_frame()
        class_object.get_coordinates()
        class_object.homo_matrix()
        class_object.warp_image()
        class_object.get_transformmed_tracks()
        class_object.plot_tracks()

    g6 = Cameras("hogni", 24032021, "2403_G6_sync")
    run(g6)

    s7 = Cameras("hogni", 24032021, "2403_S7_sync")
    run(s7)

    iph12 = Cameras("hogni", 24032021, "2403_edi_sync")
    run(iph12)


    x_list = []
    y_list = []

    for idx, i in enumerate(g6.transformed_x):
        x_list.append(i)
        y_list.append(g6.transformed_y[idx])

    for idx, i in enumerate(s7.transformed_x):
        x_list.append(i)
        y_list.append(s7.transformed_y[idx])

    for idx, i in enumerate(iph12.transformed_x):
        x_list.append(i)
        y_list.append(iph12.transformed_y[idx])

    combined = morph.show_transformed_tracker_data(x_list, y_list, g6.base_image, g6.birds_eye_view_image)
    morph.show_data(combined)