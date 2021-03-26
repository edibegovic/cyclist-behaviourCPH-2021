import trackerdf as tdf
import morph
import trajectory
import cv2

# Variables.
# ------------------------------------------

temp = []

who_is_running_this_code = "hogni"
library = "Library"
video_folder = "24032021"

file_name = "2403_edi_sync"
parent_path = f"/Users/{who_is_running_this_code}/{library}/Mobile Documents/com~apple~CloudDocs/Bachelor Project/"
parent_path_video = f"{parent_path}Videos/{video_folder}/"

tracker_path = f"{parent_path_video}Data/{file_name}/tracker_{file_name}.json"
video_path = f"{parent_path_video}Processed/{file_name}.mp4"
photo_path = f"{parent_path_video}Photos/{file_name}"
base_image = f"{parent_path}Base Image"

# Make tracker df.
# ------------------------------------------

tracker_df = tdf.create_tracker_df(tracker_path)
tracker_df = morph.cyclist_contact_coordiantes(tracker_df)
tracker_df = morph.smooth_tracks(tracker_df, 20)

# Cut df
# ------------------------------------------
#tracker_df = morph.cut_tracks_with_few_points(tracker_df, 100)

# Capture frame from video
# ------------------------------------------

src_image = morph.capture_image_from_video(video_path, base_image, file_name, 1000)


# Get points on src and dst images
# ------------------------------------------

src_image_points = morph.click_coordinates(f"{base_image}/{file_name}.jpg")
dst_image_points = morph.click_coordinates(f"{base_image}/{file_name}.jpg")

# Get homography matrix
# ------------------------------------------

homo, status = morph.find_homography_matrix(src_image, dst_image)

# Display warped image
# ------------------------------------------

warped_img = morph.warped_perspective(
    "100.jpg", "FullHD_bridge.png", homo
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
    x_list, y_list, homo, "FullHD_bridge.png"
)
morph.show_data(plotted_tracks)

# Save frame
# ------------------------------------------

def save_frame(frame_number, source, arrows=None):
    vc = cv2.VideoCapture(source)
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    rval, frame = vc.read()
    if arrows != None:
        for a in arrows:
            frame = cv2.arrowedLine(frame, a['start'], a['end'], (0,0,255), thickness=8, tipLength=0.6)
    cv2.imwrite(str(frame_number) + '.jpg', frame)

save_frame(100, video_path)

video_path
tracker_path