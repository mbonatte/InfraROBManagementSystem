import unittest
import random

import numpy as np
import pandas as pd

from convert.organization import Organization

from convert.ASFiNAG import ASFiNAG
from convert.COST_354 import COST_354

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

class TestASFiNAG(unittest.TestCase):

    def setUp(self):
        df_properties = pd.DataFrame({'Section_Name': ['road_1', 'road_2'],
                                       'Asphalt_Thickness': [3, 4],
                                       'Street_Category': ['highway', 'country_road'],
                                       'Age': ['01/01/2013', '01/01/2010'],
                                       })
        
        self.organization = ASFiNAG(df_properties)
        
        self.df_inspections = pd.DataFrame({'Section_Name': ['road_1', 'road_2'],
                                            'Date': ['01/01/2013', '01/01/2010'],
                                            'Cracking': [0, 20],
                                            'Surface_Defects': [0, 30],
                                            'Transverse_Evenness': [0, 20],
                                            'Longitudinal_Evenness': [1, 3],
                                            'Skid_Resistance': [.75, 0.6],
                                            'Bearing_Capacity': [.5, 8]
                                           })
        
        
    def test_transformation_functions(self):
        ZG_SR = np.array([0, 5, 10, 17, 30])
        result = self.organization.standardize_rutting(ZG_SR)
        np.testing.assert_array_almost_equal(result, [1,1.875,2.75,3.975,5])
        
        ZG_GR = np.array([0.75, 0.65, 0.55, 0.45, 0.35])
        result = self.organization.standardize_grip(ZG_GR)
        np.testing.assert_array_almost_equal(result, [1.499975,2.166645,2.833315,3.4999,4.9285])
        
        ZG_LE = np.array([0, 1, 2, 3, 4])
        result = self.organization.standardize_longitudinal_evenness(ZG_LE)
        np.testing.assert_array_almost_equal(result, [1,1.7778,2.5556,3.3334,4.1112])
        
        ZG_RI = np.array([0, 3, 5, 7, 10])
        result = self.organization.standardize_crack(ZG_RI)
        np.testing.assert_array_almost_equal(result, [1,2.05,2.75,3.45,4.5])
        
        ZG_OS = np.array([0, 10, 25, 35, 45])
        result = self.organization.standardize_surface_damage(ZG_OS)
        np.testing.assert_array_almost_equal(result, [1,1.875,3.1875,4.0625,4.9375])
        
        ZG_Tragf = np.array([0, 3, 6, 8, 12])
        result = self.organization.bearing_capacity_condition_index(ZG_Tragf)
        np.testing.assert_array_almost_equal(result, [1,2.05,3.1,3.8,5])
        
        age = np.array([1, 5, 10, 20, 40])
        asphalt_thickness = np.array([0.6, 1, 2, 5, 10])
        result = self.organization.age_surface_condition_index(age, asphalt_thickness)
        np.testing.assert_array_almost_equal(result, [0.13,1.33,2.83,4.03,8.23])
        result = self.organization.age_surface_condition_index(np.flip(age), asphalt_thickness)
        np.testing.assert_array_almost_equal(result, [11.83,5.83,2.83,0.88,0.04])
    
    def test_combination_functions(self):
        # safety_condition_index
        ZW_SR = np.array([1, 2, 3, 4, 5])
        ZW_GR = np.array([1, 2, 3, 4, 5])
        PI_safety_1 = self.organization.safety_condition_index(ZW_SR, ZW_GR)
        np.testing.assert_array_almost_equal(PI_safety_1, [1. , 2.1, 3.2, 4.3, 5. ])
        PI_safety_2 = self.organization.safety_condition_index(np.flip(ZW_SR), ZW_GR)
        np.testing.assert_array_almost_equal(PI_safety_2, [5,4.1,3.2,4.1,5])
        
        # confort_condition_index
        ZW_LE = np.array([1, 2, 3, 4, 5])
        ZW_OS = np.array([1, 2, 3, 4, 5])
        PI_confort_1 = self.organization.confort_condition_index(ZW_LE, ZW_OS)
        np.testing.assert_array_almost_equal(PI_confort_1, [1.002188,2.000875,3.001969,4.0035,5])
        PI_confort_2 = self.organization.confort_condition_index(np.flip(ZW_LE), ZW_OS)
        np.testing.assert_array_almost_equal(PI_confort_2, [5,4.000875,3.001969,2.0035,1.054688])
        
        # functional_condition_index
        PI_functional_1 = self.organization.functional_condition_index(PI_safety_1, PI_confort_1)
        np.testing.assert_array_almost_equal(PI_functional_1, [1.002188,2.200087,3.400197,4.60035,5.])
        PI_functional_2 = self.organization.functional_condition_index(PI_safety_2, PI_confort_1)
        np.testing.assert_array_almost_equal(PI_functional_2, [5,4.200088,3.400197,4.40035,5])
        
        # surface_structural_condition_index
        ZW_RI = np.array([1, 2, 3, 4, 5])
        ZW_OS = np.array([1, 2, 3, 4, 5])
        ZG_SR = np.array([0, 5, 10, 17, 30])
        ZG_LE = np.array([0, 1, 2, 3, 4])
        age = np.array([1, 5, 10, 20, 40])
        asphalt_thickness = np.array([0.6, 1, 2, 5, 10])
        SI_Decke_1 = self.organization.surface_structural_condition_index(ZW_RI, 
                                                                         ZW_OS, 
                                                                         ZG_SR,
                                                                         ZG_LE, 
                                                                         age, 
                                                                         asphalt_thickness)
        
        np.testing.assert_array_almost_equal(SI_Decke_1, [1.,2.1,3.2,4.3,5.4])
        SI_Decke_2 = self.organization.surface_structural_condition_index(ZW_RI, 
                                                                         np.flip(ZW_OS), 
                                                                         np.flip(ZG_SR),
                                                                         ZG_LE, 
                                                                         np.flip(age), 
                                                                         asphalt_thickness)
        
        np.testing.assert_array_almost_equal(SI_Decke_2, [5,4.1,3.2,4.1,5.])
        
        # total_structural_condition_index
        ZG_Tragf = np.array([0, 3, 6, 8, 12])
        SI_Tragf = self.organization.bearing_capacity_condition_index(ZG_Tragf)
        
        SI_gesamt_1 = self.organization.total_structural_condition_index(SI_Decke_1, SI_Tragf)
        np.testing.assert_array_almost_equal(SI_gesamt_1, [1,2.075,3.15,4.05,5.2])
        SI_gesamt_2 = self.organization.total_structural_condition_index(SI_Decke_1, np.flip(SI_Tragf))
        np.testing.assert_array_almost_equal(SI_gesamt_2, [3,2.95,3.15,3.175,3.2])
        
        # global_condition_index
        street_category = 'highway'
        GW = self.organization.global_condition_index(PI_functional_1, SI_gesamt_1, street_category)
        np.testing.assert_array_almost_equal(GW, [1.002188,2.200087,3.400197,4.60035,5])
        GW = self.organization.global_condition_index(PI_functional_2, SI_gesamt_1, street_category)
        np.testing.assert_array_almost_equal(GW, [5,4.200088,3.400197,4.40035,5])
        GW = self.organization.global_condition_index(PI_functional_1, SI_gesamt_2, street_category)
        np.testing.assert_array_almost_equal(GW, [2.67,2.6255,3.400197,4.60035,5])
        
        street_category = 'country_road'
        GW = self.organization.global_condition_index(PI_functional_1, SI_gesamt_1, street_category)
        np.testing.assert_array_almost_equal(GW, [1.002188,2.200087,3.400197,4.60035,5.])
        GW = self.organization.global_condition_index(PI_functional_2, SI_gesamt_1, street_category)
        np.testing.assert_array_almost_equal(GW, [5,4.200088,3.400197,4.40035,5])
        GW = self.organization.global_condition_index(PI_functional_1, SI_gesamt_2, street_category)
        np.testing.assert_array_almost_equal(GW, [2.4,2.36,3.400197,4.60035,5])
        
    def test_transform_performace_indicators(self):
        indicators = self.organization.single_performance_index.keys()
        df = self.organization.transform_performace_indicators(self.df_inspections)
        
        np.testing.assert_array_almost_equal(df['Cracking_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Surface_Defects_ASFiNAG'], [1,4])
        np.testing.assert_array_almost_equal(df['Transverse_Evenness_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Longitudinal_Evenness_ASFiNAG'], [2,3])
        np.testing.assert_array_almost_equal(df['Skid_Resistance_ASFiNAG'], [1,2])
        np.testing.assert_array_almost_equal(df['Bearing_Capacity_ASFiNAG'], [1,4])
        np.testing.assert_array_almost_equal(df['Safety_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Comfort_ASFiNAG'], [2,3])
        np.testing.assert_array_almost_equal(df['Functional_ASFiNAG'], [2,5])
        np.testing.assert_array_almost_equal(df['Surface_Structural_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Structural_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Global_ASFiNAG'], [2,5])
        
class TestCOST_354(unittest.TestCase):

    def setUp(self):
        df_properties = pd.DataFrame({'Section_Name': ['road_1', 'road_2'],
                                       'Asphalt_Thickness': [3, 4],
                                       'Street_Category': ['Primary', 'Secondary'],
                                       'Age': ['01/01/2013', '01/01/2010'],
                                       })
        
        self.organization = COST_354(df_properties)
        
        self.df_inspections = pd.DataFrame({'Section_Name': ['road_1', 'road_2'],
                                            'Date': ['01/01/2013', '01/01/2010'],
                                            'Cracking': [0, 20],
                                            'Surface_Defects': [0, 30],
                                            'Transverse_Evenness': [0, 20],
                                            'Longitudinal_Evenness': [1, 3],
                                            'Skid_Resistance': [.75, 0.6],
                                            'Bearing_Capacity': [.5, 8]
                                           })
        
        
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
