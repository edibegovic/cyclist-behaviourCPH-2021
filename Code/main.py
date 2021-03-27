
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
base_image = f"{parent_path}Base Image"

# ------------------------------------------

tracker_df = tdf.create_tracker_df(tracker_path)
tracker_df = morph.cyclist_contact_coordiantes(tracker_df)
tracker_df = morph.smooth_tracks(tracker_df, 20)

# Cut df
# ------------------------------------------
tracker_df = morph.cut_tracks_with_few_points(tracker_df, 100)

# Capture frame from video
# ------------------------------------------

src_image = morph.capture_image_from_video(video_path, base_image, file_name, 1000)


# Get points on src and dst images
# ------------------------------------------

src_image_points = morph.click_coordinates(f"{base_image}/{file_name}.jpg")
dst_image = morph.click_coordinates("../data/dbro_map.png")

# Get homography matrix
# ------------------------------------------

homo, status = morph.find_homography_matrix(src_image_points, dst_image_points)

# Display warped image
# ------------------------------------------

warped_img = morph.warped_perspective(
    (f"{base_image}/{file_name}.jpg"), (f"{base_image}/{birds_eye_view_image}"), homo
)
morph.show_data(warped_img)

# Plot tracks
# ------------------------------------------

x_list = []
y_list = []
for index, row in tracker_df.iterrows():
    x_list.append(row["mean_x"])
    y_list.append(row["mean_y"])

plotted_tracks = morph.transform_and_plot_tracker_data(
    x_list, y_list, homo, (f"{base_image}/{birds_eye_view_image}")
)
morph.show_data(plotted_tracks)
