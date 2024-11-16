import numpy as np

from . import organization

# TP  = Technical Parameter
# PI  = Performance Index
# CPI = Combined Performance Index
# GPI = General Performance Indicator

"""
• Longitudinal evenness; 
• Transverse evenness; 
• Macro-texture; 
• Friction; 
• Bearing Capacity; 
• Noise; 
• Air Pollution; 
• Cracking; 
• Surface defects.
"""

"""
• Safety Index; 
• Comfort Index; 
• Structural Index; 
• Environmental Index.
"""

class COST_354(organization.Organization):
    
    @staticmethod
    def transform_longitudinal_evenness(TP_IRI, number):
        #Transformation [1] was developed to create a more restrictive range
        if number == 1:
            PI_E = (0.1733 * TP_IRI**2 
                     + 0.7142 * TP_IRI
                     - 0.0316)
        elif number == 2:
            PI_E = 0.816 * TP_IRI
        PI_E = np.where(PI_E > 5, 5, PI_E)
        PI_E = np.where(PI_E < 0, 0, PI_E)
        return PI_E

    @staticmethod
    def transform_transversal_evenness(TP_TD, number):
        """
        Transformation [1] can be used for all road classes.
        Transformation [2] should only be used for motorways and primary roads.
        Transformation [3] should only be used secondary and local roads.
        """
        if number == 1:
            PI_R = (- 0.0016 * TP_TD**2 
                    + 0.2187 * TP_TD)
        elif number == 2:
            PI_R = (- 0.0015 * TP_TD**2 
                    + 0.2291 * TP_TD)
        elif number == 3:
            PI_R = (- 0.0023 * TP_TD**2 
                    + 0.2142 * TP_TD)
        return np.where(PI_R > 5, 5, PI_R)
    
    @staticmethod
    def transform_macro_texture(TP_T, street_category):
        """
        Transformation [1] is suitable for Motorway and Primary roads.
        Transformation [2] is suitable for Secondary roads.
        """
        if street_category == 'Motorway' or 'Primary':
            PI_T = 6.6 - 5.3 * TP_T 
        elif street_category == 'Secondary':
            PI_T = 7.0 - 6.9 * TP_T 
        return np.where(PI_T > 5, 5, PI_T)
    
    @staticmethod
    def transform_skid_resistance(TP_F, device):
        """
        Transformation [1] should only be used for SFC devices running at 60km/h.
        Transformation [2] should only be used for LFC devices running at 50km/h.
        """
        if device == 'SFC':
            PI_F = -17.6 * TP_F + 11.205
        elif device == 'LFC':
            PI_F = -13.875 * TP_F + 9.338
        return np.where(PI_F > 5, 5, PI_F)
    
    @staticmethod
    def transform_bearing_capacity(TP_B, device, base):
        """
        """
        if device == 'R/D':
            PI_B = 5 * (1 - TP_B)
        elif device == 'SCI_300' and base=='weak':
            PI_B = TP_B / 129
        elif device == 'SCI_300' and base=='strong':
            PI_B = TP_B / 253
        return np.where(PI_B > 5, 5, PI_B)
    
    @staticmethod
    def transform_cracking(TP_CR, street_category):
        if street_category == ('Highway' or 'Motorway'):
            PI_CR = 0.16 * TP_CR
        else:
            PI_CR = 0.1333 * TP_CR
        return np.where(PI_CR > 5, 5, PI_CR)

    @staticmethod
    def transform_surface_damage(TP_SD):
        PI_SD = 0.1333 * TP_SD
        return np.where(PI_SD > 5, 5, PI_SD)
    
    single_performance_index = {
        'Longitudinal_Evenness':'Longitudinal_Evenness',
        'Transverse_Evenness':  'Transverse_Evenness',
        'Skid_Resistance':      'Skid_Resistance',
        'Macro_Texture':        'Macro_Texture',
        'Bearing_Capacity':     'Bearing_Capacity',
        'Cracking':             'Cracking',
        'Surface_Defects':      'Surface_Defects',
    }
    
    combined_performance_index = {
        'Safety':     ['Rutting', 'Grip'],
        'Comfort':    {'Minimum': ['Longitudinal_Evenness'],
                       'Standard': ['Longitudinal_Evenness','Surface_Defects','Transverse_Evenness'],
                       'Optimum': ['Longitudinal_Evenness','Surface_Defects','Transverse_Evenness','Macro_Texture','Cracking']},
         'Functional': ['Safety', 'Comfort'],
    }
    
    worst_IC = 5
    best_IC = 0
    
    def __init__(self, properties):
        super().__init__(properties)
        # self.influence_factor = 0.2
        # self.alternative = '1'
        
        self.transformation_functions = {
            'Cracking': self.transform_cracking,
            'Surface_Defects': self.transform_surface_damage,
            'Transverse_Evenness': self.transform_transversal_evenness,
            'Longitudinal_Evenness': self.transform_longitudinal_evenness,
            'Macro_Texture': self.transform_macro_texture,
            'Skid_Resistance': self.transform_skid_resistance,
            'Bearing_Capacity': self.transform_bearing_capacity,
        }
        
        self.combination_functions = {
            # 'Safety': self.safety_performance_index,
            'Comfort': self.comfort_performance_index,
            # 'Functional': self.functional_condition_index,
            # 'Structural': self.structural_condition_index,
            # 'Bearing_Capacity': self.bearing_capacity_condition_index,
        }

    
    
##############################################################################

    def define_level(self, indicator, peformance_indices):
        levels = ['Optimum', 'Standard', 'Minimum']
        print(peformance_indices.columns)
        for level in levels:
            print(f"{level=}")
            print(f"{indicator=}")
            indicators = self.combined_performance_index[indicator][level]
            for i,ind in enumerate(indicators):
                ind += '_' + self.__class__.__name__
                indicators[i] = ind
            print(f"{indicators=}")
            if set(indicators).issubset(set(peformance_indices.columns)):
                return level
    
    def transform_weights(self):
        pass

    def get_combined_performance_index(self, indicator, peformance_indices):
        print('get_combined_performance_index')
        level = self.define_level(indicator, peformance_indices)
        print(peformance_indices)
        return self.combined_performance_index[indicator][level]

    def comfort_performance_index(self, df_inspections):
        weights = {'Longitudinal_Evenness': 1.0,
                    'Surface_Defects': 0.6,
                    'Transverse_Evenness': 0.7,
                    'Macro_Texture':0.4,
                    'Cracking':0.5}
        
        level = self.define_level('Comfort', df_inspections)
        combination = self.combined_performance_index['Comfort'][level]
        try:
            for indicator in combination:
                print(f"{indicator=}")
                weight = weights[indicator]
                indicator_index = df_inspections[indicator].astype(float)
                multiply = list(indicator_index*weight)
                #print(indicator,list(indicator_index),weight,multiply)
            return df_inspections['Longitudinal_Evenness_COST_354']
        except ValueError:
            #print(list(df_inspections['Longitudinal_Evenness_COST_354']))
            return list(df_inspections['Longitudinal_Evenness_COST_354'])