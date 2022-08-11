import os
import sys
from os.path import abspath, dirname

import numpy as np
import pandas as pd

sys.path.append(os.path.join(dirname(dirname(dirname(abspath(__file__)))), "src/model"))

from model.train import split_data

# import pprint
# pprint.pprint(sys.path)


def test_split_data():
    df = pd.DataFrame(
        {
            "PatientID": np.random.normal(size=500),
            "Pregnancies": np.random.normal(size=500),
            "PlasmaGlucose": np.random.normal(size=500),
            "DiastolicBloodPressure": np.random.normal(size=500),
            "TricepsThickness": np.random.normal(size=500),
            "SerumInsulin": np.random.normal(size=500),
            "BMI": np.random.normal(size=500),
            "DiabetesPedigree": np.random.normal(size=500),
            "Age": np.random.normal(size=500),
            "Diabetic": np.random.normal(size=500),
        }
    )
    X_train, X_test, y_train, y_test = split_data(df)

    assert X_train.shape[0] == 350
    assert X_test.shape[0] == 150
    assert y_train.shape[0] == 350
    assert y_test.shape[0] == 150
