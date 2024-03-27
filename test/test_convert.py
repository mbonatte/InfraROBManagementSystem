import unittest
import random

import numpy as np
import pandas as pd

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
               
if __name__ == '__main__':
    unittest.main()
