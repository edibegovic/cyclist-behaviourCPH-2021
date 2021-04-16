
import pandas as pd
from btrack.dataio import localizations_to_objects

def trajectory_df(df):
    """Creates df for trajectory calculation

    Parameters
    ----------
    df : Pandas df
        The pandas tracker df

    Returns
    -------
    trajectory_df : Pandas df
        A pandas df
    """
    named_df = df[["frameId", "mean_x", "mean_y"]].rename(columns={"frameId": "t", "mean_x": "x", "mean_y": "y"})
    named_df['z'] = 0.0
    return named_df

if __name__ == "__main__":
    pass
