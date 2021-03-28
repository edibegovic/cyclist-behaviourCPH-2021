
import trackerdf as tdf
import morph
import trajectory
import cv2

# Variables
# ------------------------------------------

temp = []

who_is_running_this_code = "hogni"

video_folder = "24032021"
file_name = "2403_edi_sync"

parent_path = f"/Users/{user}/library/Mobile Documents/com~apple~CloudDocs/Bachelor Project/Videos/{video_folder}/"

tracker_path = f"{parent_path}Data/{file_name}/tracker_{file_name}.json"
video_path = f"{parent_path}Processed/{file_name}.mp4"
photo_path = f"{parent_path}Photos/{file_name}"
map_path = f"../data/dbro_map.png"

# ------------------------------------------

tracker_df = tdf.create_tracker_df(tracker_path)
tracker_df = morph.cyclist_contact_coordiantes(tracker_df)
tracker_df = morph.smooth_tracks(tracker_df, 20)

# Remove all tracks with 100 points or fewers
# ------------------------------------------
tracker_df = morph.cut_tracks_with_few_points(tracker_df, 50)

# Capture frame from video
# ------------------------------------------

video_frame = morph.get_frame(video_path, 800)


# Get points on src and dst images
# ------------------------------------------


src_image = morph.click_coordinates(video_frame)
dst_image = morph.click_coordinates(map_path)

# Get homography matrix
# ------------------------------------------

homo, _ = morph.find_homography_matrix(src_image, dst_image)

# Display warped image
# ------------------------------------------

warped_img = morph.warped_perspective(
    (video_frame, "../data/dbro_map.png", homo
)
morph.show_data(warped_img)

# Plot tracks
# ------------------------------------------

x_list = []
tracks = tracker_df[:1400]
transformed_tracks = morph.transform_points(tracks, homo)

# Plot on MAP
plot = morph.get_cv2_point_plot(transformed_tracks, map_path)
morph.show_data(plot)

# Plot on VIDEO (FRAME)
plot = morph.get_cv2_point_plot(tracks, video_frame)
morph.show_data(plot)

