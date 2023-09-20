from . import organization

import numpy as np

class ASFiNAG(organization.Organization): 
    """
    ASFiNAG has five single performance indicators:
        Griffigkeit -> Skid_Resistance
        Spurrinnen -> Transverse_Evenness
        Längsebenheit -> Longitudinal_Evenness
        Oberflächenschäden -> Surface_Defects
        Risse -> Cracking
    
    The general formulation of a normalization function is as follows:
        ZW_i = 1,0 + X . ZG_i    [1,0 ≤ ZW_i ≤ 5,0]
    """
    single_performance_index = {'Cracking':             'Cracking',
                                'Surface_Defects':      'Surface_Defects',
                                'Transverse_Evenness':  'Transverse_Evenness',
                                'Longitudinal_Evenness':'Longitudinal_Evenness',
                                'Skid_Resistance':      'Skid_Resistance',
                                'Bearing_Capacity':      'Bearing_Capacity',
                                }
    combined_performance_index = {'Safety':     ['Transverse_Evenness',
                                                 'Skid_Resistance'],
                                'Comfort':    ['Longitudinal_Evenness',
                                               'Surface_Defects'],
                                'Functional': ['Safety',
                                               'Comfort'],
                                'Surface_Structural': ['Cracking',
                                                       'Surface_Defects',
                                                       'Skid_Resistance',
                                                       'Longitudinal_Evenness',],
                                'Structural': ['Surface_Structural',
                                               'Bearing_Capacity',],
                                'Global': ['Functional',
                                           'Structural',],
                                  }
                                
    
    
    def __init__(self,
                 properties):
        self.properties = properties
        self.asphalt_thickness = properties['Asphalt_Thickness']
        self.street_category = properties['Street_Category']
        self.street_age = properties['Age']
        
        self.transformation_functions = {'Cracking': self.standardize_crack,
                                        'Surface_Defects': self.standardize_surface_damage,
                                        'Transverse_Evenness': self.standardize_rutting,
                                        'Longitudinal_Evenness': self.standardize_longitudinal_evenness,
                                        'Skid_Resistance': self.standardize_grip,
                                        'Bearing_Capacity': self.bearing_capacity_condition_index,
                                        }
        self.combination_functions = {
                                    'Safety': self.safety_condition_index,
                                    'Comfort': self.confort_condition_index,
                                    'Functional': self.functional_condition_index,
                                    'Surface_Structural': self.surface_structural_condition_index,
                                    'Structural': self.total_structural_condition_index,
                                    'Global': self.global_condition_index,
                                    }
    
    def convert_indicator(self, indicator):
        return self.standardize_function[indicator]

##############################################################################

    def standardize_rutting(self, ZG_SR):
        ZW_SR = 1 + 0.175 * ZG_SR
        ZW_SR = np.where(ZW_SR > 5, 5, ZW_SR)
        return ZW_SR
    
    def standardize_grip(self, ZG_GR):
        ZW_GR = np.where(ZG_GR <= 0.45,
                         9.9286 - 14.286 * ZG_GR,
                         6.5 - 6.6667 * ZG_GR)
        ZW_GR = np.where(ZW_GR > 5, 5, ZW_GR)
        return ZW_GR
        
    def standardize_longitudinal_evenness(self, ZG_LE):
        ZW_LE = 1 + 0.7778 * ZG_LE
        ZW_LE = np.where(ZW_LE > 5, 5, ZW_LE)
        return ZW_LE
    
    def standardize_crack(self, ZG_RI):
        ZW_RI = 1 + 0.35 * ZG_RI
        ZW_RI = np.where(ZW_RI > 5, 5, ZW_RI)
        return ZW_RI
    
    def standardize_surface_damage(self, ZG_OS):
        ZW_OS = 1 + 0.0875 * ZG_OS
        ZW_OS = np.where(ZW_OS > 5, 5, ZW_OS)
        return ZW_OS
        
    def bearing_capacity_condition_index(self, ZG_Tragf):
        """
        chapter 4.2.3 and chapter 4.2.4.
        
        ZG_Tragf -> Theoretical load-bearing capacity
        SI_Tragf = SI_BC
        """
        SI_Tragf = 1 + 0.35 * ZG_Tragf
        return np.where(SI_Tragf > 5, 5, SI_Tragf)
    
    def age_surface_condition_index(self, age, asphalt_thickness):
        """
        ZW_{Alter,As} -> Condition value age for asphalt pavements
        Alter_{Decke} -> Age of top layer [years]
        
        How to calculate the age? What is the time reference?
        """
        #Values in cm        
        if asphalt_thickness > 2:
            return 0.21 * age - 0.17
        if asphalt_thickness <= 2:
            return 0.30 * age - 0.17
    
##############################################################################
    
    def safety_condition_index(self, ZW_SR, ZW_GR):
        """
        ZW_SR -> rutting
        ZW_GR -> grip (or skid)
        """
        PI_safety = max(ZW_SR, ZW_GR) + 0.1*min(ZW_SR, ZW_GR) - 0.1
        return np.where(PI_safety > 5, 5, PI_safety)
    
    def confort_condition_index(self, ZW_LE, ZW_OS):
        """
        ZW_LE -> longitudinal evenness
        ZW_OS -> surface damage
        """
        PI_confort = (max(ZW_LE, 1 + 0.0021875 * ZW_OS**2)
                      + 0.1*min(ZW_LE, 1 + 0.0021875 * ZW_OS**2) 
                      - 0.1)
        return np.where(PI_confort > 5, 5, PI_confort)
    
    def functional_condition_index(self, PI_safety, PI_confort):
        PI_functional = (max(PI_safety, PI_confort) 
                         + 0.1*min(PI_safety, PI_confort) 
                         - 0.1)
        return np.where(PI_functional > 5, 5, PI_functional)
    

    def surface_structural_condition_index(self, ZW_RI, ZW_OS, ZG_SR, 
                                   ZG_LE, age, asphalt_thickness):
        """
        ZW_RI -> cracking
        ZW_OS -> surface damage
        ZG_SR -> rutting
        ZG_LE -> longitudinal_evenness
        
        SI_{Decke} = SI_AS
        """
        ZW_AlterAS = self.age_surface_condition_index(age, asphalt_thickness)
        
        return max(max(ZW_RI, ZW_OS) + 0.1*min(ZW_RI, ZW_OS)-0.1,
                   max(min(1+0.00010938*ZG_SR**3, 5),
                       min(1+0.03840988*ZG_LE**3, 5)),
                   min(0.08*ZW_RI+0.61, 0.85)*ZW_AlterAS
                   )
    
    
    def theoretical_load_bearing_capacity(self):
        """
        K_Tragf -> Calibration factor state variable carrying capacity
        K_Risse -> Calibration factor cracks (from condition recording)
        J_akt -> current year of the analysis or investigation [year]
        J_rechn -> calculated superstructure year [year]
        VBI
        n -> Assessment period in years (motorways and expressways = 30 years, otherwise 20 years)
        """
        K_Tragf = K_Risse
        
        value = K_Tragf * np.exp(-3.6017 + 0.1*(J_akt - J_rechn) + np.ln(J_akt - J_rechn + 0.01))
        
        if VBI >= 0.7:
            return value / (VBI * n / 20)
        else:
            return value / VBI
    
    def total_structural_condition_index(self, SI_Decke, SI_Tragf):
        """
        SI_{Decke} = SI_AS
        SI_Tragf = SI_BC
        
        DickeDecke -> thickness of the top layer [cm]
        DickeGebSchichten -> Total thickness of all bound layers [cm]
        """
        Dicke_Decke = 2
        Dicke_GebSchiten = 2
        
        if Dicke_Decke <= Dicke_GebSchiten:
            return ((SI_Decke * Dicke_Decke + SI_Tragf * Dicke_GebSchiten)
                    / (Dicke_Decke + Dicke_GebSchiten))
        
    def global_condition_index(self, GI, SI, street_category):
        """
        GI = CSI
        """
        WGI = 1
        if street_category == 'highway':
            WSI = 0.89
        if street_category == 'country_road':
            WSI = 0.8
        return max(WGI*GI, WSI*SI)
    
    
##############################################################################
    
    def combine_indicator(self, indicator,df_inspections,to_suffix=True):
        suffix = ''
        if to_suffix:
            suffix = '_ASFiNAG'
        
        function = self.combination_functions[indicator]
        
        if indicator == 'Safety':
            return np.array(list(map(function, 
                            df_inspections['Transverse_Evenness'+suffix],
                            df_inspections['Skid_Resistance'+suffix])))
        
        if indicator == 'Comfort':
            return np.array(list(map(function, 
                            df_inspections['Longitudinal_Evenness'+suffix],
                            df_inspections['Surface_Defects'+suffix])))
        
        if indicator == 'Functional':
            return np.array(list(map(function, 
                            df_inspections['Safety'+suffix],
                            df_inspections['Comfort'+suffix])))
        
        if indicator == 'Surface_Structural':
            final_result = []
            
            for index, row in df_inspections.iterrows():
                filtered_by_name = self.properties[self.properties['Section_Name'] == row['Section_Name']]
                
                age = filtered_by_name['Age'].values[0]
                asphalt_thickness = filtered_by_name['Asphalt_Thickness'].values[0]
                
                from datetime import datetime
                
                time_diff = datetime.strptime(age, "%d/%m/%Y") - datetime.strptime(row['Date'], "%d/%m/%Y")
                time_diff = time_diff.days/365
            
                step_result = function(row['Cracking'],
                                       row['Surface_Defects'+suffix],
                                       row['Transverse_Evenness'],
                                       row['Longitudinal_Evenness'+suffix],
                                       time_diff,
                                       asphalt_thickness)
                
                final_result.append(step_result)
            
            return np.array(final_result)
            
            return np.array(list(map(function, 
                            df_inspections['Cracking'],
                            df_inspections['Surface_Defects'+suffix],
                            df_inspections['Transverse_Evenness'],
                            df_inspections['Longitudinal_Evenness'+suffix],
                            self.street_age)))
        
        if indicator == 'Structural':
            return np.array(list(map(function, 
                            df_inspections['Surface_Structural'+suffix],
                            df_inspections['Bearing_Capacity'+suffix])))
        
        if indicator == 'Global':
            final_result = []
            for index, row in df_inspections.iterrows():
                filtered_by_name = self.properties[self.properties['Section_Name'] == row['Section_Name']]
                street_category = filtered_by_name['Street_Category'].values[0]
                
                result = function(row['Functional'+suffix],
                                  row['Structural'+suffix],
                                  street_category)

                final_result.append(result)
            return np.array(final_result)