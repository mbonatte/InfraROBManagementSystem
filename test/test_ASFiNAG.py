import unittest
import random

import numpy as np

from InfraROBManagementSystem.convert.ASFiNAG import ASFiNAG

class TestASFiNAG(unittest.TestCase):

    def setUp(self):        
        properties = {
            'name': 'road_1',
            'asphalt_surface_thickness': 4,
            'total_pavement_thickness': 12,
            'street_category': 'highway',
            'age': 1,
            }
        
        self.organization = ASFiNAG(properties)
        
        self.inspections = {
            'Date': ['01/01/2014', '01/01/2011'],
            'Cracking': [0, 20],
            'Surface_Defects': [0, 30],
            'Transverse_Evenness': [0, 20],
            'Longitudinal_Evenness': [1, 3],
            'Skid_Resistance': [.75, 0.6],
            'Bearing_Capacity': [.5, 8]
            }
        
    def test_standardize_functions_single_value(self):    
        ZG_SR = np.array(10)
        ZW_SR = self.organization.standardize_transverse_evenness(ZG_SR)
        self.assertAlmostEqual(ZW_SR, 2.75)
        
        ZG_GR = np.array(0.55)
        ZW_GR = self.organization.standardize_skid_resistance(ZG_GR)
        self.assertAlmostEqual(ZW_GR, 2.833315)
    
    def test_standardize_functions_array(self):
        ZG_SR = np.array([0, 5, 10, 17, 30])
        ZW_SR = self.organization.standardize_transverse_evenness(ZG_SR)
        np.testing.assert_array_almost_equal(ZW_SR, [1,1.875,2.75,3.975,5])
        
        ZG_GR = np.array([0.75, 0.65, 0.55, 0.45, 0.35])
        ZW_GR = self.organization.standardize_skid_resistance(ZG_GR)
        np.testing.assert_array_almost_equal(ZW_GR, [1.499975,2.166645,2.833315,3.4999,4.9285])
        
        ZG_LE = np.array([0, 1, 2, 3, 4])
        ZW_LE = self.organization.standardize_longitudinal_evenness(ZG_LE)
        np.testing.assert_array_almost_equal(ZW_LE, [1,1.7778,2.5556,3.3334,4.1112])
        
        ZG_RI = np.array([0, 3, 5, 7, 10])
        result = self.organization.standardize_crack(ZG_RI)
        np.testing.assert_array_almost_equal(result, [1,2.05,2.75,3.45,4.5])
        
        ZG_OS = np.array([0, 10, 25, 35, 45])
        result = self.organization.standardize_surface_damage(ZG_OS)
        np.testing.assert_array_almost_equal(result, [1,1.875,3.1875,4.0625,4.9375])
        
        ZG_Tragf = np.array([0, 3, 6, 8, 12])
        result = self.organization.standardize_bearing_capacity(ZG_Tragf)
        np.testing.assert_array_almost_equal(result, [1,2.05,3.1,3.8,5])
    
    def test_standardize_age(self):    
        age = np.array(10)
        asphalt_thickness = np.array(2)
        result = self.organization.standardize_age_of_asphalt_structure(age, asphalt_thickness)
        self.assertAlmostEqual(result, 2.83)
        
        age = np.array([1, 5, 10, 20, 40])
        asphalt_thickness = np.array([0.6, 1, 2, 5, 10])
        result = self.organization.standardize_age_of_asphalt_structure(age, asphalt_thickness)
        np.testing.assert_array_almost_equal(result, [1,1.33,2.83,4.03,5])
        
        result = self.organization.standardize_age_of_asphalt_structure(np.flip(age), asphalt_thickness)
        np.testing.assert_array_almost_equal(result, [5,5,2.83,1,1])
    
    ############################################################
    
    def test_comfort_safety_functional(self):
        # safety_condition_index
        ZW_SR = 3
        ZW_GR = 3
        
        PI_safety = self.organization.calculate_safety_index(ZW_SR, ZW_GR)
        self.assertAlmostEqual(PI_safety, 3.2)
        
        ZW_SR = np.array([1, 2, 3, 4, 5])
        ZW_GR = np.array([1, 2, 3, 4, 5])
        
        PI_safety_1 = self.organization.calculate_safety_index(ZW_SR, ZW_GR)
        np.testing.assert_array_almost_equal(PI_safety_1, [1. , 2.1, 3.2, 4.3, 5. ])
        
        PI_safety_2 = self.organization.calculate_safety_index(np.flip(ZW_SR), ZW_GR)
        np.testing.assert_array_almost_equal(PI_safety_2, [5,4.1,3.2,4.1,5])
        
        # comfort_condition_index
        ZW_LE = 3
        ZW_OS = 3
        
        PI_confort = self.organization.calculate_comfort_index(ZW_LE, ZW_OS)
        self.assertAlmostEqual(PI_confort, 3.114286, delta=1e-5)
        
        ZW_LE = np.array([1, 2, 3, 4, 5])
        ZW_OS = np.array([1, 2, 3, 4, 5])
        
        PI_confort_1 = self.organization.calculate_comfort_index(ZW_LE, ZW_OS)
        np.testing.assert_array_almost_equal(PI_confort_1, [1,2.028571,3.114286,4.257143,5])
        
        PI_confort_2 = self.organization.calculate_comfort_index(np.flip(ZW_LE), ZW_OS)
        np.testing.assert_array_almost_equal(PI_confort_2, [5,4.028571,3.114286,3.671429,5])
        
        PI_confort_3 = self.organization.calculate_comfort_index(ZW_LE, np.flip(ZW_OS))
        np.testing.assert_array_almost_equal(PI_confort_3, [5,3.671429,3.114286,4.028571,5])
        
        # functional_condition_index
        PI_safety = 3
        PI_confort = 3
        
        PI_functional = self.organization.calculate_functional_index(PI_safety, PI_confort)
        self.assertAlmostEqual(PI_functional, 3.2)
        
        PI_functional_1 = self.organization.calculate_functional_index(PI_safety_1, PI_confort_1)
        np.testing.assert_array_almost_equal(PI_functional_1, [1,2.202857,3.411429,4.625714,5])
        
        PI_functional_2 = self.organization.calculate_functional_index(PI_safety_2, PI_confort_1)
        np.testing.assert_array_almost_equal(PI_functional_2, [5,4.202857,3.411429,4.567143,5])
    
    def test_surface_structural_index(self):
        ZG_SR = np.array([0, 5, 10, 17, 30])
        ZG_LE = np.array([0, 1, 2, 3, 4])
        
        ZW_SR = self.organization.standardize_transverse_evenness(ZG_SR)
        ZW_LE = self.organization.standardize_longitudinal_evenness(ZG_LE)
        
        ZW_RI = np.array([1, 2, 3, 4, 5])
        ZW_OS = np.array([1, 2, 3, 4, 5])
        
        
        age = np.array([1, 5, 10, 20, 40])
        asphalt_thickness = np.array([0.6, 1, 2, 5, 10])
        
        SI_Decke_1 = self.organization.calculate_surface_structural_index(ZW_RI, 
                                                                         ZW_OS, 
                                                                         ZW_SR,
                                                                         ZW_LE, 
                                                                         age, 
                                                                         asphalt_thickness)
        
        np.testing.assert_array_almost_equal(SI_Decke_1, [1.,2.1,3.2,4.3,5])
        
        SI_Decke_2 = self.organization.calculate_surface_structural_index(ZW_RI, 
                                                                         np.flip(ZW_OS), 
                                                                         np.flip(ZW_SR),
                                                                         ZW_LE, 
                                                                         np.flip(age), 
                                                                         asphalt_thickness)
        
        np.testing.assert_array_almost_equal(SI_Decke_2, [5,4.1,3.2,4.1,5.])
        
    def test_structural_index(self):
        SI_Decke_1 = np.array([1.,2.1,3.2,4.3,5])
        ZG_Tragf = np.array([0, 3, 6, 8, 12])
        SI_Tragf = self.organization.standardize_bearing_capacity(ZG_Tragf)
        
        asphalt_surface_thickness = 3 #cm
        total_pavement_thickness = 10 #cm
        
        SI_gesamt_1 = self.organization.calculate_structural_index(SI_Decke_1, SI_Tragf,
                                                                   asphalt_surface_thickness, total_pavement_thickness)
        np.testing.assert_array_almost_equal(SI_gesamt_1, [1,2.061538,3.123077,3.915385,5])
        
        SI_gesamt_2 = self.organization.calculate_structural_index(SI_Decke_1, np.flip(SI_Tragf),
                                                                   asphalt_surface_thickness, total_pavement_thickness)
        np.testing.assert_array_almost_equal(SI_gesamt_2, [4.076923,3.407692,3.123077,2.569231,1.923077])
    
    def test_global_index(self):
        GI = np.array([1, 2, 3, 4, 5])
        SI = np.array([1, 2, 3, 4, 5])
        
        street_category = 'highway'
        
        GW = self.organization.calculate_global_index(GI, SI, street_category)
        np.testing.assert_array_almost_equal(GW, [1, 2, 3, 4, 5])
        
        GW = self.organization.calculate_global_index(np.flip(GI), SI, street_category)
        np.testing.assert_array_almost_equal(GW, [5.  , 4.  , 3.  , 3.56, 4.45])
        
        GW = self.organization.calculate_global_index(GI, np.flip(SI), street_category)
        np.testing.assert_array_almost_equal(GW, [4.45, 3.56, 3.  , 4.  , 5.  ])
        
        street_category = 'country_road'
        
        GW = self.organization.calculate_global_index(GI, SI, street_category)
        np.testing.assert_array_almost_equal(GW, [1., 2., 3., 4., 5.])
        
        GW = self.organization.calculate_global_index(np.flip(GI), SI, street_category)
        np.testing.assert_array_almost_equal(GW, [5. , 4. , 3. , 3.2, 4. ])
        
        GW = self.organization.calculate_global_index(GI, np.flip(SI), street_category)
        np.testing.assert_array_almost_equal(GW, [4. , 3.2, 3. , 4. , 5. ])
    
    ############################################################
    
    def test_safety_from_ASFiNAG_database(self):
        pass
    
    def test_comfort_from_ASFiNAG_database(self):
        ZW_LE_ASFiNAG = np.array([2.367, 1.167,1.144,1.100,1.100])
        ZW_OS_ASFiNAG = np.array([1.005, 1.729,2.948,5.000,3.241])
        PI_comfort_ASFiNAG = np.array([2.367, 1.182,2.098,5.000,2.445])
        
        PI_comfort = self.organization.calculate_comfort_index(ZW_LE_ASFiNAG, ZW_OS_ASFiNAG)
        np.testing.assert_array_almost_equal(PI_comfort, PI_comfort_ASFiNAG, decimal=1)
    
    def test_functional_from_ASFiNAG_database(self):
        PI_safety_ASFiNAG = [1.862,1.921,1.929,1.822,2.110,1.860,2.534,2.125,2.196,1.869,1.809,1.817,1.809]
        PI_confort_ASFiNAG = [1.811, 1.644, 2.544, 2.522, 2.411, 2.800, 3.644, 3.089, 4.673, 2.200, 2.024, 1.989, 4.653]
        PI_functional_ASFiNAG = [1.943,1.985,2.637,2.604,2.522,2.886,3.797,3.202,4.793,2.287,2.105,2.071,4.734]        
        
        PI_functional = self.organization.calculate_functional_index(PI_safety_ASFiNAG, PI_confort_ASFiNAG)
        
        np.testing.assert_array_almost_equal(PI_functional_ASFiNAG, PI_functional, decimal=3)
    
    def test_surface_structural_index_from_ASFiNAG_database(self):
        pass
    
    def test_structural_index_from_ASFiNAG_database(self):
        pass
    
    def test_global_index_from_ASFiNAG_database(self):
        GI_ASFiNAG = np.array([2.974,2.017,1.992,2.013,1.998,2.018,1.992,2.550])
        SI_ASFiNAG = np.array([3.854,5.000,1.471,1.456,2.582,1.459,1.405,5.000])
        GW_ASFiNAG = np.array([3.430,4.450,1.992,2.013,2.298,2.018,1.992,4.450])
        
        street_category = 'highway'
        
        GW = self.organization.calculate_global_index(GI_ASFiNAG, SI_ASFiNAG, street_category)
        np.testing.assert_array_almost_equal(GW, GW_ASFiNAG, decimal=4)
    
    ############################################################
    
    def test_combine_indicator_simple(self):
        # Setup a simple DataFrame for testing
        df_inspections = {
            'Transverse_Evenness_ASFiNAG': [1, 2, 3],
            'Skid_Resistance_ASFiNAG': [1, 3, 5]
        }
        indicator = 'Safety'
        columns = ['Transverse_Evenness', 'Skid_Resistance']
        properties_needed = []
        suffix = '_ASFiNAG'
        expected = np.array([1, 3.1, 5])
        result = self.organization._combine_indicator(df_inspections, indicator, columns, properties_needed, suffix)
        np.testing.assert_array_equal(result, expected, "Combination function did not produce expected results")
    
    def test_combine_indicator(self):
        # Setup a simple DataFrame for testing
        df_inspections = {
            'Transverse_Evenness_ASFiNAG': [1, 2, 3],
            'Skid_Resistance_ASFiNAG': [1, 3, 5],
            'Cracking_ASFiNAG': [1, 3, 5],
            'Surface_Defects_ASFiNAG': [1, 2, 3],
            'Longitudinal_Evenness_ASFiNAG': [1, 3, 5],
            'Bearing_Capacity_ASFiNAG': [1, 3, 5],
        }
        
        df_inspections['Section_Name'] =['road_1', 'road_2', 'road_1']
        df_inspections['Date'] = ['01/01/2014', '01/01/2011', '01/01/2013']
        
        indicator = 'Safety'
        expected = np.array([1, 3.1, 5])
        result = self.organization.combine_indicator(indicator, df_inspections)
        np.testing.assert_array_equal(result, expected)
        
        indicator = 'Comfort'
        expected = np.array([1, 3.028571, 5])
        result = self.organization.combine_indicator(indicator, df_inspections)
        np.testing.assert_allclose(result, expected, rtol=1e-05)
        
        df_inspections['Safety_ASFiNAG'] = [1, 3 ,5]
        df_inspections['Comfort_ASFiNAG'] = [1, 2 ,4]
        indicator = 'Functional'
        expected = np.array([1, 3.1, 5])
        result = self.organization.combine_indicator(indicator, df_inspections)
        np.testing.assert_allclose(result, expected, rtol=1e-05)
        
        indicator = 'Surface_Structural'
        expected = np.array([1, 3.1, 5])
        result = self.organization.combine_indicator(indicator, df_inspections)
        np.testing.assert_allclose(result, expected, rtol=1e-05)
        
        df_inspections['Surface_Structural_ASFiNAG'] = [1, 3 ,5]
        indicator = 'Structural'
        expected = np.array([1, 3, 5])
        result = self.organization.combine_indicator(indicator, df_inspections)
        np.testing.assert_allclose(result, expected, rtol=1e-05)
        
        df_inspections['Functional_ASFiNAG'] = [1, 3 ,5]
        df_inspections['Structural_ASFiNAG'] = [1, 2 ,4]
        indicator = 'Global'
        expected = np.array([1, 3, 5])
        result = self.organization.combine_indicator(indicator, df_inspections)
        np.testing.assert_allclose(result, expected, rtol=1e-05)
        
    def test_transform_performance_indicators(self):
        indicators = self.organization.single_performance_index.keys()
        
        df = self.organization.transform_performance_indicators(self.inspections)
        
        np.testing.assert_array_almost_equal(df['Cracking_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Surface_Defects_ASFiNAG'], [1,4])
        np.testing.assert_array_almost_equal(df['Transverse_Evenness_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Longitudinal_Evenness_ASFiNAG'], [2,3])
        np.testing.assert_array_almost_equal(df['Skid_Resistance_ASFiNAG'], [1,2])
        np.testing.assert_array_almost_equal(df['Bearing_Capacity_ASFiNAG'], [1,4])
        
        np.testing.assert_array_almost_equal(df['Safety_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Comfort_ASFiNAG'], [2,4])
        np.testing.assert_array_almost_equal(df['Functional_ASFiNAG'], [2,5])
        np.testing.assert_array_almost_equal(df['Surface_Structural_ASFiNAG'], [1,5])
        np.testing.assert_array_almost_equal(df['Structural_ASFiNAG'], [1,4])
        
        np.testing.assert_array_almost_equal(df['Global_ASFiNAG'], [2,5])
        
       
if __name__ == '__main__':
    unittest.main()
