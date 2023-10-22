import numpy as np
import requests
def collect_example_dataset():
    """
    Collect Data set of CSGO
    Data contains ADR vs Rating of a Player
    :return dataset obtained from link
    """
    response = requests.get(
        "https://raw.githubusercontent.com/yashLadha/The_Math_of_Intelligence/"
        "master/Week1/ADRvsRating.csv"
    )
    lines = response.text.splitlines()[1:]
    data = []
    for data_point in lines:
        data.append(data_point.split(','))
    return np.array(data).astype(np.float64)


