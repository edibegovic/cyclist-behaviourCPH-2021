# import importlib
# importlib.reload(project)
import tracker
import project

class Camera:
    def __init__(self, user, video_folder, file_name):
        self.user = user
        self.video_folder = video_folder
        self.file_name = file_name

        self.parent_path = f"/Users/{self.user}/Library/Mobile Documents/com~apple~CloudDocs/Bachelor Project/"
        self.parent_path_video = f"{self.parent_path}Videos/{self.video_folder}/"
        self.tracker_path = f"{self.parent_path_video}Data/{self.file_name}/tracker_{self.file_name}.json"
        self.video_path = f"{self.parent_path_video}Processed/{self.file_name}.mp4"
        self.photo_path = f"{self.parent_path_video}Photos/{self.file_name}"
        self.base_image = f"{self.parent_path}Base Image"
        self.map_path = f"{self.base_image}/FullHD_bridge.png"


g6 = Camera("hogni", 24032021, "2403_G6_sync")

# ------------------------------------------

tracker_df = tracker.create_tracker_df(g6.tracker_path)
tracker_df = project.cyclist_contact_coordiantes(tracker_df)
tracker_df = project.smooth_tracks(tracker_df, 20)

# Remove all tracks with 100 points or fewers
# ------------------------------------------

tracker_df = project.cut_tracks_with_few_points(tracker_df, 50)

# Capture frame from video
# ------------------------------------------

video_frame = project.get_frame(g6.video_path, 800)

# Get points on src and dst images
# ------------------------------------------

src_image = project.click_coordinates(video_frame)
dst_image = project.click_coordinates(g6.map_path)

# Get homography matrix
# ------------------------------------------

homo, _ = project.find_homography_matrix(src_image, dst_image)

# Display warped image
# ------------------------------------------

warped_img = project.warped_perspective(video_frame, g6.map_path, homo)
project.show_data(warped_img)

# Plot tracks
# ------------------------------------------

x_list = []
tracks = tracker_df[:1400]
transformed_tracks = project.transform_points(tracks, homo)

# Plot on MAP
plot = project.get_cv2_point_plot(transformed_tracks, g6.map_path)
project.show_data(plot)

# Plot on VIDEO (FRAME)
plot = project.get_cv2_point_plot(tracks, video_frame)
project.show_data(plot)