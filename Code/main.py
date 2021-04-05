import trackerdf as tdf
import morph
import ml

class Cameras:
    def __init__(
        self,
        who_is_running_this_code,
        video_folder,
        file_name,
        frame_number=1000,
        smooth_factor=20,
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
        morph.capture_image_from_video(self.video_path, self.base_image, self.file_name, self.frame_number)
        src_image_points = morph.click_coordinates(f"{self.base_image}/{self.file_name}.jpg")
        dst_image_points = morph.click_coordinates(f"{self.base_image}/{self.birds_eye_view_image}")
        self.homo, status = morph.find_homography_matrix(src_image_points, dst_image_points)
        self.warped_img = morph.warped_perspective((f"{self.base_image}/{self.file_name}.jpg"),(f"{self.base_image}/{self.birds_eye_view_image}"),self.homo)
        morph.show_data(self.warped_img)
        self.tracker_df = morph.transform_points(self.tracker_df, self.homo)
        self.tracker_img = morph.get_cv2_point_plot(self.tracker_df, (f"{self.base_image}/{self.birds_eye_view_image}"))
        morph.show_data(self.tracker_img)
        self.n_clusters, self.labels, self.uniqueid, self.model = ml.run_all(g6.tracker_df, [6])

if __name__ == "__main__":

    g6 = Cameras("hogni", 24032021, "2403_G6_sync")
    g6.run()

    s7 = Cameras("hogni", 24032021, "2403_S7_sync")
    s7.run()

    iph12 = Cameras("hogni", 24032021, "2403_edi_sync")
    iph12.run()