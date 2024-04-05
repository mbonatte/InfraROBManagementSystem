import unittest
import random

import numpy as np

from InfraROBManagementSystem.convert.organization import Organization

from InfraROBManagementSystem.convert.ASFiNAG import ASFiNAG
from InfraROBManagementSystem.convert.COST_354 import COST_354

class newOrganization(Organization):
    def single_performance_index(self):
        pass
    def combined_performance_index(self):
        pass

class TestOrganization(unittest.TestCase):

    def setUp(self):
        self.organization = newOrganization({})
        
    def test_standardize_values(self):
        indicator_values = np.array([0.5,
                                     2.0,
                                     3.0,
                                     4.0,
                                     4.9])
        
        result = self.organization.standardize_values(indicator_values)
        
        np.testing.assert_array_almost_equal(result, [1,2,3,4,5])
        
    def test_add_suffix(self):
        columns = ['Column1', 'Column2']
        suffix = '_Test'
        expected = ['Column1_Test', 'Column2_Test']
        result = self.organization._add_suffix(columns, suffix)
        self.assertEqual(result, expected, "Suffixes were not correctly added to column names")
        
    def test_calculate_dates_difference_in_years(self):
        start_date_str = "01/01/2020"
        end_date_str = "01/01/2022"
        expected = 2
        result = self.organization._calculate_dates_difference_in_years(start_date_str, end_date_str)
        self.assertEqual(result, expected, "Date difference in years was not calculated correctly")
               
if __name__ == '__main__':
    unittest.main()
