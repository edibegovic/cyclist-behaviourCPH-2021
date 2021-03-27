import pandas as pd


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

    trajectory_df = df[["frameId", "mean_x", "mean_y"]].copy()
    trajectory_df["z"] = 0
    trajectory_df["state"] = 0
    trajectory_df["label"] = 0
    trajectory_df.columns = ["t", "x", "y", "z", "state", "label"]
    trajectory_df = trajectory_df.astype(float)
    return trajectory_df


if __name__ == "__main__":
    pass
