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
g6 = camera_class.Camera("hogni", 24032021, "short_g6")

# Create df
# ------------------------------------------
# ODC
g6.tracker_df = tracker.create_tracker_df(g6.tracker_path)
# YOLOv5 - load
g6.tracker_df = pd.read_pickle("short_g6_yolov5x6.pickle")

# Road contact point and smooth trajectories
# ------------------------------------------
g6.tracker_df = project.cyclist_contact_coordiantes(g6.tracker_df)
g6.tracker_df = project.smooth_tracks(g6.tracker_df, 20)

# Remove all tracks with less that n points
# ------------------------------------------
g6.tracker_df = project.cut_tracks_with_few_points(g6.tracker_df, 50)

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
project.show_data(warped_img)

# Plot tracks
# ------------------------------------------
g6.tracker_df = project.transform_points(g6.tracker_df, homo)

# Connect points with UniqueID
# ------------------------------------------
g6.tracker_df = connect.set_unique_id(g6.tracker_df, max_age=30, min_hits=1, iou_threshold=0.15)

# Plot on MAP
# ------------------------------------------
plot = project.get_cv2_point_plot(g6.tracker_df, g6.map_path)
project.show_data(plot)

# Plot on VIDEO (FRAME)
# ------------------------------------------
plot = project.get_cv2_point_plot(g6.tracker_df, video_frame)
project.show_data(plot)

# df to CSV
# ------------------------------------------
g6.tracker_df.to_csv("short_g6_yolov5x6_id.csv")