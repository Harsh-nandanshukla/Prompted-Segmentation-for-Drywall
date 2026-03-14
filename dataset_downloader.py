
from roboflow import Roboflow

rf = Roboflow(api_key="wl90gcKdVpFcKspjA5Ug")

# Drywall dataset
project1 = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
dataset1 = project1.version(1).download("coco")

# Cracks dataset (your fork)
project2 = rf.workspace("harshs-workspace-pitin").project("cracks-3ii36-ie0rs")
dataset2 = project2.version(1).download("coco")

print("Datasets downloaded successfully")