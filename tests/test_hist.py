import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from plot_data import calc_grid_hist


class TestMapCalc(unittest.TestCase):

    def setUp(self):
        start_date = datetime.strptime('2020-10-05','%Y-%m-%d')
        self.df = pd.DataFrame()
        self.df['LATITUDE'] = np.array([1.5,1.5,2.1,2.2,2.3])
        self.df['LONGITUDE'] = np.array([1.8,2,1.1,1.1,1.1])
        self.df['DATETIME'] = pd.date_range(start_date,periods=5).tolist()
        self.box = [1,3,1,3]

    def test_calcgridhist(self):
        res = 1
        x_edges = np.arange(self.box[0] - res, self.box[1] + res, res)
        y_edges = np.arange(self.box[2] - res, self.box[3] + res, res)
        hist_array = calc_grid_hist(x_coords=self.df['LONGITUDE'], y_coords=self.df['LATITUDE'], time=self.df['DATETIME'], x_edges=x_edges, y_edges=y_edges, res=1)
        expected_value = {'x': np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 3.5,
        3.5, 3.5, 3.5]), 'y': np.array([0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5, 0.5,
        1.5, 2.5, 3.5]), 'h': np.array([np.nan, np.nan, np.nan, np.nan, np.nan,  1.,  3., np.nan, np.nan,  1., np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan])}
        np.testing.assert_equal(expected_value['x'],hist_array['x'])
        np.testing.assert_equal(expected_value['y'],hist_array['y'])
        np.testing.assert_equal(expected_value['h'],hist_array['h'])


