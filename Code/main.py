import cyclist
import pandas as pd

def join_df(df_list):
    return pd.concat(df_list, ignore_index=True).sort_values("frame_id").reset_index(drop=True)

if __name__ == "__main__":
    g6 = cyclist.Camera("2403_g6_sync", "g6")
    g6.read_pkl("2403_g6_sync_yolov5x6_resized_corrected")
    g6.unique_id(max_age=90, min_hits=1, iou_threshold=0.10, save_load = "load")
    g6.cyclist_contact_coordiantes()
    g6.smooth_tracks(20)
    src = g6.click_coordinates(g6.frame, dst = "src", type = "load")
    dst = g6.click_coordinates(g6.map_path, dst = "dst", type = "load")
    g6.find_homography_matrix(src, dst)
    g6.transform_points()

    s7 = cyclist.Camera("2403_s7_sync", "s7")
    s7.read_pkl("2403_s7_sync_yolov5x6_resized_corrected")
    s7.unique_id(max_age=90, min_hits=1, iou_threshold=0.10, save_load = "load")
    s7.cyclist_contact_coordiantes()
    s7.smooth_tracks(20)
    src = s7.click_coordinates(s7.frame, dst = "src", type = "load")
    dst = s7.click_coordinates(s7.map_path, dst = "dst", type = "load")
    s7.find_homography_matrix(src, dst)
    s7.transform_points()

    remove_line = g6.click_coordinates(g6.map_path, dst = 0, type = "line")
    g6.remove_point_line(remove_line, "below")
    s7.remove_point_line(remove_line, "above")

    joined = cyclist.Camera("joined_resized_corrected", "joined")
    joined.tracker_df = join_df([g6.tracker_df, s7.tracker_df])
    joined.new_bbox(15)
    joined.df_format()
    joined.unique_id(max_age=90, min_hits=1, iou_threshold=0.10, save_load = "load")

    joined.add_bearing()
    joined.smooth_bearings(10, type = "bearing")
    joined.add_color(type="rainbow")

    plot = joined.plot_object(joined.tracker_df, joined.map_path)
    joined.show_data("Warped img", plot)

    joined.tracker_df.to_csv(f"CSV/{joined.file_name}_unique_id_{joined.max_age}_{joined.iou_threshold}.csv")