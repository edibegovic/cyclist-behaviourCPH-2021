import pandas as pd
import json


def create_tracker_df(tracker_file_path):
    """Creates tracker df

    Parameters
    ----------
    tracker_file_path : json
        The file location of the tracker.json

    Returns
    -------
    Pandas df
        a pandas df of the tracker.json file
    """
    with open(tracker_file_path) as f:
        tracker = json.load(f)
    tracker_flattend = pd.json_normalize(
        tracker, record_path="objects", meta=["frameId"]
    ).rename(columns={"id": "UniqueID"})
    tracker_flattend = tracker_flattend.drop_duplicates(
        subset=["UniqueID", "frameId"], keep="first"
    )
    return tracker_flattend


def save_tracker_df(tracker_df, save_path):
    """Save tracker df

    Parameters
    ----------
    tracker_df : Pandas df
        The Pandas df to save

    save_path : str
        Path where to save df

    """
    tracker_df.to_csv(save_path)
    return "Saved"


def load_tracker_df(tracker_df_path):
    """Loads tracker df

    Parameters
    ----------
    tracker_df_path : csv
        The file location of the saved tracker.csv file

    Returns
    -------
    Pandas df
        a pandas df of the tracker.csv file
    """
    return pd.read_csv(tracker_df_path)


if __name__ == "__main__":
    pass
