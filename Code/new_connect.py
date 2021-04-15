

def connect(tracks):
    grouped_bikes = tracks[tracks["name"]] == "bike"].groupby("unique_id")
    grouped_df = tracks[tracks["name"]] == "person"].groupby("frame_id")

    for name, group in grouped_bikes:
        




if __name__ == "__main__":