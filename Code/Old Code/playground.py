import trackerdf as tdf
import morph
import trajectory
import btrack
from btrack.dataio import import_CSV
from collections import OrderedDict
import pandas as pd

import importlib

importlib.reload(trajectory)

t_df = trajectory.trajectory_df(tracker_df)
t_df.to_csv("tracker_df.csv")

objects = import_CSV("tracker_df.csv")
with btrack.BayesianTracker() as tracker:

    # configure the tracker using a config file
    tracker.configure_from_file("cell_config.json")

    # append the objects to be tracked
    tracker.append(objects)

    # set the volume (Z axis volume is set very large for 2D data)
    tracker.volume = ((0, 1920), (0, 1028), (-1e5, 1e5))
    # track them (in interactive mode)
    tracker.track_interactive(step_size=100)

    # generate hypotheses and run the global optimizer
    tracker.optimize()

    # get the tracks as a python list
    tracks = tracker.tracks


df = pd.DataFrame()
for track in tracks:
    temp_df = pd.DataFrame({"unique_id": track.ID, "mean_x": track.x, "mean_y": track.y})
    df.append(temp_df, ignore_index=True)

print(df)
