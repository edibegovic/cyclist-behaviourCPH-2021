import numpy as np
import pandas as pd
import cv2

temp = []
img = 0


def cyclist_contact_coordiantes(df):
    df["y"] = df["y"] + df["h"] / 2
    return df


def smooth_tracks(df, smoothing_factor):
    df_x = (
        df.groupby("unique_id")["x"]
        .rolling(smoothing_factor, min_periods=1)
        .mean()
        .to_frame(name="smooth_x")
        .droplevel("unique_id")
    )

    df_y = (
        df.groupby("unique_id")["y"]
        .rolling(smoothing_factor, min_periods=1)
        .mean()
        .to_frame(name="smooth_y")
        .droplevel("unique_id")
    )

    df["x"] = df_x
    df["y"] = df_y
    return df


def cut_tracks_with_few_points(df, n):
    return df[df.groupby("unique_id")["unique_id"].transform("size") > n]


def find_homography_matrix(source_list, destination_list):
    coordiantes_on_source = np.array(source_list)
    coordiantes_on_destination = np.array(destination_list)

    return cv2.findHomography(coordiantes_on_source, coordiantes_on_destination)


def warped_perspective(src, dst, matrix):
    if isinstance(src, str):
        source_image = cv2.imread(src)
    else:
        source_image = src.copy()

    if isinstance(dst, str):
        destination_image = cv2.imread(dst)
    else:
        destination_image = dst.copy()

    return cv2.warpPerspective(
        source_image, matrix, (destination_image.shape[1], destination_image.shape[0])
    )


def transform_points(points, matrix):
    trans_points = points.copy()
    transformed_x = []
    transformed_y = []

    # Apply transformation for each point
    for _, row in points.iterrows():
        point = (row["x"], row["y"])

        transformed_x.append(
            int(
                round(
                    (matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2])
                    / (
                        (
                            matrix[2][0] * point[0]
                            + matrix[2][1] * point[1]
                            + matrix[2][2]
                        )
                    )
                )
            )
        )

        transformed_y.append(
            int(
                round(
                    (matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2])
                    / (
                        (
                            matrix[2][0] * point[0]
                            + matrix[2][1] * point[1]
                            + matrix[2][2]
                        )
                    )
                )
            )
        )

    trans_points.drop(columns=["x", "y"])
    trans_points["x"] = transformed_x
    trans_points["y"] = transformed_y
    return trans_points


def plot_object(tracker_df, dst_image):
    if isinstance(dst_image, str):
        image = cv2.imread(dst_image)
    else:
        image = dst_image.copy()

    grouped = tracker_df.groupby("unique_id")
    for _, group in grouped:
        xy = []
        colour_list = []
        for count, (_, row) in enumerate(group.iterrows()):
            xy.append((row["x"], row["y"]))
            colour_list.append((0, 0, 255))  # row["colour"])
            if len(xy) > 1:
                cv2.line(image, xy[count - 1], xy[count], colour_list[count], 3)
    return image


def show_data(name, cv2_object):
    cv2.imshow(name, cv2_object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click_event(event, x, y, flags, params):
    global temp
    if event == cv2.EVENT_LBUTTONDOWN:

        temp.append([x, y])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x, y), 10, (200, 90, 255), -1)
        cv2.putText(img, str(len(temp)), (x + 5, y - 5), font, 2, (255, 255, 255), 5)
        cv2.imshow("image", img)


def click_coordinates(image):
    global temp
    global img
    if temp:
        temp = []
        img = 0

    if isinstance(image, str):
        img = cv2.imread(image, 1)
    else:
        img = image

    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return temp


def capture_image_from_video(video_path, base_image, file_name, frame_number):
    vc = cv2.VideoCapture(video_path)
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    rval, frame_number = vc.read()
    cv2.imwrite(f"{base_image}/{file_name}.jpg", frame_number)
    return f"Frame {frame_number} Saved"


def get_frame(video_path, frame_number):
    vc = cv2.VideoCapture(video_path)
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    rval, frame = vc.read()
    return frame


def join_df(df1, df2):
    data = {
        "frame_id": [],
        "unique_id": [],
        "x": [],
        "y": [],
        "camera": [],
    }
    for _, row in df1.iterrows():
        data["frame_id"].append(row["frame_id"])
        data["unique_id"].append(row["unique_id"])
        data["x"].append(row["x"])
        data["y"].append(row["y"])
        data["camera"].append(row["camera"])
    for _, row in df2.iterrows():
        data["frame_id"].append(row["frame_id"])
        data["unique_id"].append(row["unique_id"])
        data["x"].append(row["x"])
        data["y"].append(row["y"])
        data["camera"].append(row["camera"])
    new_df = pd.DataFrame(data=data)
    return new_df


def add_camera(df, camera):
    df["camera"] = camera
    return df


if __name__ == "__main__":
    pass
