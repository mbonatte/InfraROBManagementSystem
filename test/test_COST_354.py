import unittest
import random

import numpy as np

from InfraROBManagementSystem.organization.COST_354 import COST_354
      
class TestCOST_354(unittest.TestCase):

    def setUp(self):
        df_properties = {'Section_Name': ['road_1', 'road_2'],
                                       'Asphalt_Thickness': [3, 4],
                                       'Street_Category': ['Primary', 'Secondary'],
                                       'Age': ['01/01/2013', '01/01/2010'],
                                       }
        
        self.organization = COST_354(df_properties)
        
        self.df_inspections = {'Section_Name': ['road_1', 'road_2'],
                                            'Date': ['01/01/2013', '01/01/2010'],
                                            'Cracking': [0, 20],
                                            'Surface_Defects': [0, 30],
                                            'Transverse_Evenness': [0, 20],
                                            'Longitudinal_Evenness': [1, 3],
                                            'Skid_Resistance': [.75, 0.6],
                                            'Bearing_Capacity': [.5, 8]
                                           }
        
        
    def test_transformation_functions(self):
        TP_IRI = np.array([0, 0.5, 1.5, 2.2, 3.0, 3.5, 4])
        PI_E = self.organization.transform_longitudinal_evenness(TP_IRI)
        np.testing.assert_array_almost_equal(PI_E, [0,0.368825,1.429625,2.378412,3.6707,4.591025,5])
        
        TP_TD = np.array([0, 3, 7, 12, 20, 25, 35])
        PI_R = self.organization.transform_transversal_evenness(TP_TD)
        np.testing.assert_array_almost_equal(PI_R, [0,0.6738,1.5302,2.5332,3.982,4.79,5])
        
        TP_T = np.array([1.1, 0.95, .75, .55, .35])
        PI_T = self.organization.transform_macro_texture(TP_T, 'Primary')
        np.testing.assert_array_almost_equal(PI_T, [0.77 , 1.565, 2.625, 3.685, 4.745])
        
        TP_F = np.array([0.6, 0.55, .5, .45, .39])
        PI_F = self.organization.transform_skid_resistance(TP_F, 'SFC')
        np.testing.assert_array_almost_equal(PI_F, [0.645, 1.525, 2.405, 3.285, 4.341])
        
        TP_B = np.array([70, 200, 300, 450, 600])
        PI_B = self.organization.transform_bearing_capacity(TP_B, 'SCI_300', 'weak')
        np.testing.assert_array_almost_equal(PI_B, [0.542636, 1.550388, 2.325581, 3.488372, 4.651163])
        
        TP_CR = np.array([5, 10, 18, 27, 35])
        PI_CR = self.organization.transform_cracking(TP_CR, 'Highway')
        np.testing.assert_array_almost_equal(PI_CR, [0.8 , 1.6 , 2.88, 4.32, 5])
        
        TP_SD = np.array([5, 10, 18, 27, 30])
        PI_SD = self.organization.transform_surface_damage(TP_SD)
        np.testing.assert_array_almost_equal(PI_SD, [0.6665, 1.333 , 2.3994, 3.5991, 3.999])
      
if __name__ == '__main__':
    unittest.main()
