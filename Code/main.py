import trackerdf as tdf
import morph

# Variables.
# ------------------------------------------

temp = []

who_is_running_this_code = "hogni"
library = "Library"
video_folder = "16032021"

file_name = "hogni_30fps_highup_16032010"
parent_path = f"/Users/{who_is_running_this_code}/{library}/Mobile Documents/com~apple~CloudDocs/Bachelor Project/Videos/{video_folder}/"

tracker_path = f"{parent_path}Data/{file_name}/tracker_{file_name}.json"
video_path = f"{parent_path}Processed/{file_name}.mp4"
photo_path = f"{parent_path}Photos/{file_name}"
db_path = f"{parent_path}Data/{file_name}/db_{file_name}.csv"

# Make tracker df.
# ------------------------------------------

tracker_df = tdf.create_tracker_df(tracker_path)
tracker_df = morph.cyclist_contact_coordiantes(tracker_df)
tracker_df = morph.smooth_tracks(tracker_df, 20)

# Get points on src and dst images
# ------------------------------------------

src_image = morph.click_coordinates("25.jpg")
dst_image = morph.click_coordinates("Screenshot 2021-03-16 at 21.54.45.png")

# Get homography matrix
# ------------------------------------------

homo, status = morph.find_homography_matrix(src_image, dst_image)

# Display warped image
# ------------------------------------------

warped_img = morph.warped_perspective(
    "25.jpg", "Screenshot 2021-03-16 at 21.54.45.png", homo
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
    x_list, y_list, homo, "Screenshot 2021-03-16 at 21.54.45.png"
)
morph.show_data(plotted_tracks)
