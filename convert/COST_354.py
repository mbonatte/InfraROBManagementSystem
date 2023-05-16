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
    single_performance_index = {
                                'Longitudinal_Evenness':'Longitudinal_Evenness',
                                'Transverse_Evenness':  'Transverse_Evenness',
                                'Skid_Resistance':      'Skid_Resistance',
                                'Macro_Texture':        'Macro_Texture',
                                'Bearing_Capacity':     'Bearing_Capacity',
                                'Cracking':             'Cracking',
                                'Surface_Defects':      'Surface_Defects',
                                }
    combined_performance_index = {#'Safety':     ['Rutting',
                                  #               'Grip'],
                                  # 'Comfort':    {'Minimum': ['Longitudinal_Evenness'],
                                  #                'Standard': ['Longitudinal_Evenness',
                                  #                             'Surface_Defects',
                                  #                             'Transverse_Evenness'],
                                  #                'Optimum': ['Longitudinal_Evenness',
                                  #                             'Surface_Defects',
                                  #                             'Transverse_Evenness',
                                  #                             'Macro_Texture',
                                  #                             'Cracking']},
                                  #  'Functional': ['Safety',
                                  #                 'Comfort'],
                                  }
    def __init__(self, properties):
        super().__init__(properties)
        self.p = 0.2
        self.alternative = '1'
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
                                    #'Safety': self.safety_performance_index,
                                    'Comfort': self.comfort_performance_index,
                                    # 'Functional': self.functional_condition_index,
                                    # 'Structural': self.structural_condition_index,
                                    # 'Bearing_Capacity': self.bearing_capacity_condition_index,
                                    }
    
##############################################################################

    def transform_longitudinal_evenness(self, TP_IRI, number=1):
        #Transformation [1] was developed to create a more restrictive range
        if number == 1:
            PI_E = (0.1733 * TP_IRI**2 
                     + 0.7142 * TP_IRI
                     - 0.0316)
        elif number == 2:
            PI_E = 0.816 * TP_IRI
        return np.where(PI_E > 5, 5, PI_E)

    def transform_transversal_evenness(self, TP_TD, number=2):
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
    
    def transform_macro_texture(self, TP_T):
        """
        Transformation [1] is suitable for Motorway and Primary roads.
        Transformation [2] is suitable for Secondary roads.
        """
        if self.properties['Street_Category'] == 'Motorway' or 'Primary':
            PI_F = 6.6 - 5.3 * TP_T 
        elif self.properties['Street_Category'] == 'Secondary':
            PI_F = 7.0 - 6.9 * TP_T 
        return np.where(PI_F > 5, 5, PI_F)
    
    def transform_skid_resistance(self, TP_F, device='SFC'):
        """
        Transformation [1] should only be used for SFC devices running at 60km/h.
        Transformation [2] should only be used for LFC devices running at 50km/h.
        """
        if device == 'SFC':
            PI_F = -17.6 * TP_F + 11.205
        elif device == 'LFC':
            PI_F = -13.875 * TP_F + 9.338
        return np.where(PI_F > 5, 5, PI_F)
    
    def transform_bearing_capacity(self, TP_B, device='SCI_300', base='weak'):
        """
        """
        if device == 'R/D':
            PI_B = 5 * (1 - TP_B)
        elif device == 'SCI_300' and base=='weak':
            PI_B = TP_B / 129
        elif device == 'SCI_300' and base=='strong':
            PI_B = TP_B / 253
        return np.where(PI_B > 5, 5, PI_B)
    
    def transform_cracking(self, TP_CR):
        if self.properties['Street_Category'] == ('Highway' or 'Motorway'):
            PI_CR = 0.16 * TP_CR
        else:
            PI_CR = 0.1333 * TP_CR
        return np.where(PI_CR > 5, 5, PI_CR)

    def transform_surface_damage(self, TP_SD):
        PI_SD = 0.1333 * TP_SD
        return np.where(PI_SD > 5, 5, PI_SD)
    
##############################################################################

    def define_level(self, indicator, peformance_indices):
        levels = ['Optimum', 'Standard', 'Minimum']
        for level in levels:
            indicators = self.combined_performance_index[indicator][level]
            for i,indicator in enumerate(indicators):
                indicator += '_' + self.__class__.__name__
                indicators[i] = indicator
            if set(indicators).issubset(set(peformance_indices.columns)):
                return level
    
    def transform_weights(self):
        pass

    def get_combined_performance_index(self, indicator, peformance_indices):
        level = self.define_level(indicator, peformance_indices)
        return self.combined_performance_index[indicator][level]

    def comfort_performance_index(self, df_inspections):
        weights = {'Longitudinal_Evenness_COST_354': 1.0,
                    'Surface_Defects_COST_354': 0.6,
                    'Transverse_Evenness_COST_354': 0.7,
                    'Macro_Texture_COST_354':0.4,
                    'Cracking_COST_354':0.5}
        level = self.define_level('Comfort', df_inspections)
        combination = self.combined_performance_index['Comfort'][level]
        try:
            for indicator in combination:
                #print(indicator)
                weight = weights[indicator]
                indicator_index = df_inspections[indicator].astype(float)
                multiply = list(indicator_index*weight)
                #print(indicator,list(indicator_index),weight,multiply)
            return df_inspections['Longitudinal_Evenness_COST_354']
        except ValueError:
            #print(list(df_inspections['Longitudinal_Evenness_COST_354']))
            return list(df_inspections['Longitudinal_Evenness_COST_354'])