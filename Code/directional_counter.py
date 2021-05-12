
import cyclist
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


g6 = cyclist.Camera(24032021, "2403_g6_sync", "g6")
g6.tracker_df = pd.read_pickle("./2403_g6_sync_yolov5x6_corrected_unique_id.pickle")

g6.unique_id(max_age=90, min_hits=1, iou_threshold=0.15, save_load = "load")
g6.cyclist_contact_coordiantes()
g6.frame = "Data/24032021/Data/States/corrected_g6.jpg"
g6.smooth_tracks(20)
src = g6.click_coordinates(g6.frame, dst = "src", type = "load")
dst = g6.click_coordinates(g6.map_path, dst = "dst", type = "load")
g6.find_homography_matrix(src, dst)
g6.transform_points()


# -----------------------------------------------------------------
# COUNTER LINE FUNCITON
# -----------------------------------------------------------------

# df: DataFrame 
#   containing: (unique_id, x, y)
#
# polygon: String
#   containing: "M..L..L..L..Z"
#
# line: ((x1, y1), (x2, y2))
#   line segment defined by two points


df = g6.tracker_df

def directional_counter(df, polygon_str, line=None):
    coordinates_raw = [coor.replace("M", "").replace("Z", "").split(",") for coor in polygon_str.split("L")]
    coordinates = [(float(a), float(b)) for a, b in coordinates_raw]
    polygon = Polygon(coordinates)
    is_inside = lambda row: polygon.contains(Point(row['x'], row['y']))
    points_inside = df[df.apply(is_inside, axis=1)]
    incidents = points_inside.groupby('unique_id').first()

    a, b = line
    isLeft = lambda point: (((b[0] - a[0])*(point['y'] - a[1]) - (b[1] - a[1])*(point['x'] - a[0])) < 0)

    incidents["behind"] = df[df['unique_id'].isin(list(incidents.index))].groupby('unique_id').first().apply(isLeft, axis=1)
    return len(incidents[incidents['behind']])


# Edi
# polygon_string = "M651.6000000000003,584.4L692.4000000000002,560.4L735.6000000000003,543.6L764.4000000000002,589.1999999999999L798.0000000000002,642L817.2000000000002,661.1999999999999L757.2000000000002,704.4L752.4000000000002,711.6Z"
# line = ((677, 618), (800, 560))

# Høgni
polygon_str =  "M985.2000000000003,387.59999999999997L1050.0000000000002,346.8L1105.2000000000003,322.8L1210.8000000000002,265.2L1270.8000000000002,226.79999999999998L1350.0000000000002,198L1374.0000000000002,210L1402.8000000000002,241.2L1434.0000000000002,325.2L1371.6000000000001,382.8L1321.2,421.2L1170.0000000000002,498L1093.2000000000003,541.1999999999999L1076.4000000000003,548.4Z"
line = ((970, 350), (1334, 157))

directional_counter(df, polygon_str, line)
