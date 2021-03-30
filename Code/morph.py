import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

temp = []
img = 0


def cyclist_contact_coordiantes(df):
    """Calculates cyclist plane contact point

    Parameters
    ----------
    df : Pandas df
        The pandas df with x and y columns

    Returns
    -------
    Pandas df
        a pandas df with the adjusted x and y
    """
    df["y"] = df["y"] + df["h"] / 2
    # df["x"] = df["x"] + df["w"] / 2
    return df


def smooth_tracks(df, smoothing_factor):
    """Smooth tracked paths

    Parameters
    ----------
    df : Pandas df
        The pandas df with x and y columns

    smoothing_factor : int
        Factor to smooth tracker lines by

    Returns
    -------
    Pandas df
        a pandas df with the adjusted x and y
    """
    df_ = (
        df.groupby("UniqueID")["x"]
        .rolling(smoothing_factor, min_periods=1)
        .mean()
        .to_frame(name="mean_x")
        .droplevel("UniqueID")
    )
    df = df.join(df_)
    df_ = (
        df.groupby("UniqueID")["y"]
        .rolling(smoothing_factor, min_periods=1)
        .mean()
        .to_frame(name="mean_y")
        .droplevel("UniqueID")
    )
    df = df.join(df_)
    return df


def cut_tracks_with_few_points(df, n):
    """Smooth tracked paths

    Parameters
    ----------
    df : Pandas df
        The pandas tracker df

    n : int
        Cuts tracks with less than n paths

    Returns
    -------
    Pandas df
        A cut pandas df
    """
    return df[df.groupby("UniqueID")["UniqueID"].transform("size") > n]


def find_homography_matrix(source_list, destination_list):
    """Finds Homography matrix

    Parameters
    ----------
    source_list : list of lists
        Points on source images

    destination_list : list of lists
        Points on source images

    Returns
    -------
    Tuple (matrix, status)
    """
    coordiantes_on_source = np.array(source_list)
    coordiantes_on_destination = np.array(destination_list)

    return cv2.findHomography(coordiantes_on_source, coordiantes_on_destination)


def warped_perspective(src, dst, matrix):
    """Warps source image

    Parameters
    ----------
    src : str
        Path to source image

    dst : str
        Path to destination image

    matrix : np array
        Homography matrix

    Returns
    -------
    CV2 warped object
    """
    source_image = cv2.imread(src)

    if isinstance(dst, str):
        destination_image = cv2.imread(dst)
    else:
        destination_image = dst

    return cv2.warpPerspective(
        source_image, matrix, (destination_image.shape[1], destination_image.shape[0])
    )


def transform_points(points, matrix):
    """Transforms tracker data and plots on CV2 object from view_transformed_picture function

    Parameters
    ----------
    points : pd.DataFrame
    Contains rastor 2D coordinates (x, y)

    matrix : (3, 3) numpy array
    Homography matrix for projection

    Returns
    -------
    pd.DataFrame
    Transformed coordinates
    """
    trans_points = points.copy()
    transformed_x = []
    transformed_y = []

    # Apply transformation for each point
    for _, row in points.iterrows():
        point = (row["x"], row["y"])

        transformed_x.append(
            (matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2])
            / ((matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2]))
        )

        transformed_y.append(
            (matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2])
            / ((matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2]))
        )

    trans_points.drop(columns=["x", "y"])
    trans_points["x"] = transformed_x
    trans_points["y"] = transformed_y
    return trans_points


def get_cv2_point_plot(points, dst_image):

    if isinstance(dst_image, str):
        image = cv2.imread(dst_image)
    else:
        image = dst_image.copy()

    colors = [
        (0, 0, 0),
        (225, 0, 0),
        (0, 225, 0),
        (0, 0, 225),
        (225, 225, 0),
        (0, 225, 225),
        (225, 0, 225),
        (255, 255, 255),
    ]
    for _, row in points.iterrows():
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = colors[int(row["UniqueID"]) % 8]

        x, y = int(row["x"]), int(row["y"])

        cv2.circle(image, (x, y), 5, color, -1)

        # Add timestamp at first and last apperance
        # if not np.isnan(row["start_time"]):
        #     cv2.putText(
        #         image,
        #         "{:.0f}".format(row["start_time"]),
        #         (x + 9, y - 35),
        #         font,
        #         1,
        #         color,
        #         2,
        #     )

    return image


def show_data(cv2_object):
    """Display image.

    Parameters
    ----------
    cv2_object : cv2 object
        CV2 image object
    """
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2_object)
    plt.show()


def click_event(event, x, y, flags, params):
    """Needed for click coordiantes"""
    global temp
    if event == cv2.EVENT_LBUTTONDOWN:

        temp.append([x, y])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x, y), 10, (200, 90, 255), -1)
        cv2.putText(img, str(len(temp)), (x + 5, y - 5), font, 2, (255, 255, 255), 5)
        cv2.imshow("image", img)


def click_coordinates(image):
    """Display image with plotted tracker data

    Parameters
    ----------
    img_path : str
        Path to image

    Returns
    -------
    List of lists of coordinates
    """
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
    return temp


def capture_image_from_video(video_path, base_image, file_name, frame_number):
    """Save frame from video

    Parameters
    ----------
    video_path : str
        Path to video

    base_image : str
        Path to save image to

    file_name : str
        File name of video file

    frame_number : int
        Frame to save

    Returns
    -------
    str
    """
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


if __name__ == "__main__":
    pass
