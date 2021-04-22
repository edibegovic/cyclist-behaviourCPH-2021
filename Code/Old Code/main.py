# import importlib
# importlib.reload(project)
import tracker
import project
import pandas as pd
import connect
import camera_class

# To process the video files go to
# ------------------------------------------
# https://colab.research.google.com/drive/13NJxj5pvzDIDWQ5H9GO_bYGqg2dEx5Ey?usp=sharing

# Create class object
# ------------------------------------------
g6 = camera_class.Camera("hogni", 24032021, "2403_g6_sync")

# Create df
# ------------------------------------------

# ODC
# tracker_df = tracker.create_tracker_df(g6.tracker_path)
# YOLOv5 - load
tracker_df = pd.read_pickle("2403_g6_sync_yolov5x6.pickle")

# Connect points with UniqueID
# ------------------------------------------
tracker_df = connect.set_unique_id(
    tracker_df, max_age=30, min_hits=1, iou_threshold=0.15
)

# Road contact point and smooth trajectories
# ------------------------------------------
tracker_df = project.cyclist_contact_coordiantes(tracker_df)
tracker_df = project.smooth_tracks(tracker_df, 20)

# Remove all tracks with less that n points
# ------------------------------------------
tracker_df = project.cut_tracks_with_few_points(tracker_df, 50)

# Capture frame from video
# ------------------------------------------
video_frame = project.get_frame(g6.video_path, 1000)

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
project.show_data("Warped projection", warped_img)

# Transform tracks
# ------------------------------------------
tracker_df = project.transform_points(tracker_df, homo)

# Plot on MAP
# ------------------------------------------
plot = project.plot_object(tracker_df, g6.map_path)
project.show_data("Tracks projection", plot)

# df to CSV
# ------------------------------------------
df.to_csv("df_joined.csv")
tracker_df_2 = pd.read_csv("short_s7_yolov5x6_id.csv")

df = pd.concat([tracker_df, tracker_df_2], ignore_index=True)
df[:100]
