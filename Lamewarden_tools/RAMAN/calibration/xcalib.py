# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
# from sklearn.externals import joblib
import joblib


# functions
class Calibrator:
    def __init__(self, orig_anchors=None, ref_anchors=None):
        try:
            self.orig_anchors = np.array(orig_anchors)[:,0].reshape(-1,1)
            self.ref_anchors = np.array(ref_anchors)[:,0].reshape(-1,1)
            self.fit()
        except IndexError:
            self.orig_anchors = orig_anchors
            self.ref_anchors = ref_anchors
            self.calib_model = None

    def fit(self):
        for degree in range(1,15):
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            print(self.orig_anchors, self.ref_anchors)
            model.fit(self.orig_anchors, self.ref_anchors)
            pred = model.predict(self.orig_anchors)
            poly_mse = mean_squared_error(self.ref_anchors, pred)
            print(f"Degree: {degree}, MSE: {poly_mse}")
            if poly_mse < 0.00001: #we don't want to get polynom of 6th degree if we can get the same result with 2nd degree
                self.calib_model = model
                print(poly_mse)
                break

    def transform(self, data):
            calib_data = data.copy()
            calib_data.iloc[:,0] = self.calib_model.predict(np.array(data.iloc[:,0]).reshape(-1, 1))
            return calib_data
            
    def save_model(self, address):
         if hasattr(self, 'calib_model'):
              joblib.dump(self.calib_model, address)

    def load_model(self, address):
        try:
            self.calib_model = joblib.load(address)
        except FileNotFoundError:
             pass

