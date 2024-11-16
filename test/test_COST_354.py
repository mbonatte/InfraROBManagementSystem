import unittest

import numpy as np
import pandas as pd

from InfraROBManagementSystem.organization.COST_354 import COST_354
      
class TestCOST_354(unittest.TestCase):

    def setUp(self):
        df_properties = {'Section_Name': ['road_1'],
                                       'Asphalt_Thickness': [3],
                                       'Street_Category': ['Primary'],
                                       'Age': ['01/01/2013'],
                                       }
        
        self.organization = COST_354(df_properties)
        
        self.df_inspections = {
            'Section_Name': ['road_1', 'road_1', 'road_1', 'road_1', 'road_1'],
            'Date': ['01/01/2013', '01/01/2010', '01/01/2010', '01/01/2010', '01/01/2010'],
            'Cracking': np.array([5, 10, 18, 27, 35]),
            'Macro_Texture': np.array([1.1, 0.95, .75, .55, .35]),
            'Surface_Defects': np.array([5, 10, 18, 27, 30]),
            'Transverse_Evenness': np.array([0, 3, 12, 20, 35]),
            'Longitudinal_Evenness': np.array([0, 0.5, 2.2, 3.0, 4]),
            'Skid_Resistance': np.array([0.6, 0.55, .5, .45, .39]),
            'Bearing_Capacity': np.array([70, 200, 300, 450, 600])
        }
        
        
    def test_transformation_functions(self):
        TP_IRI = self.df_inspections['Longitudinal_Evenness']
        PI_E = self.organization.transform_longitudinal_evenness(TP_IRI, 1)
        np.testing.assert_array_almost_equal(PI_E, [0,0.368825,2.378412,3.6707,5])
        
        TP_TD = self.df_inspections['Transverse_Evenness']
        PI_R = self.organization.transform_transversal_evenness(TP_TD, 2)
        np.testing.assert_array_almost_equal(PI_R, [0,0.6738,2.5332,3.982,5])
        
        TP_T = self.df_inspections['Macro_Texture']
        PI_T = self.organization.transform_macro_texture(TP_T, 'Primary')
        np.testing.assert_array_almost_equal(PI_T, [0.77 , 1.565, 2.625, 3.685, 4.745])
        
        TP_F = self.df_inspections['Skid_Resistance']
        PI_F = self.organization.transform_skid_resistance(TP_F, 'SFC')
        np.testing.assert_array_almost_equal(PI_F, [0.645, 1.525, 2.405, 3.285, 4.341])
        
        TP_B = self.df_inspections['Bearing_Capacity']
        PI_B = self.organization.transform_bearing_capacity(TP_B, 'SCI_300', 'weak')
        np.testing.assert_array_almost_equal(PI_B, [0.542636, 1.550388, 2.325581, 3.488372, 4.651163])
        
        TP_CR = self.df_inspections['Cracking']
        PI_CR = self.organization.transform_cracking(TP_CR, 'Highway')
        np.testing.assert_array_almost_equal(PI_CR, [0.8 , 1.6 , 2.88, 4.32, 5])
        
        TP_SD = self.df_inspections['Surface_Defects']
        PI_SD = self.organization.transform_surface_damage(TP_SD)
        np.testing.assert_array_almost_equal(PI_SD, [0.6665, 1.333 , 2.3994, 3.5991, 3.999])

    def test_get_combined_performance_index(self):        
        indicators_prediction = pd.DataFrame(self.df_inspections)
        self.organization.transform_performance_indicator('Cracking',indicators_prediction)
        
    #     # result = self.organization.get_combined_performance_index(
    #     #     'Comfort',
    #     #     indicators_prediction)

if __name__ == '__main__':
    unittest.main()
