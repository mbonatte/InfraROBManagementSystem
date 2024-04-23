from typing import List
from datetime import datetime

import numpy as np

from .organization import Organization


class ASFiNAG(Organization):
    """
    Represents the ASFiNAG organization with methods to standardize performance indicators.
    Inherits from the abstract Organization class and implements ASFiNAG-specific methodologies
    for calculating and combining performance indicators related to road quality and safety.

    Attributes:
        asphalt_surface_thickness (float): Thickness of the asphalt in centimeters.
        street_category (str): Category of the street (e.g., 'Highway').
        street_age (int): Age of the street in years.
        properties (pd.DataFrame): Inherits from Organization, representing additional properties.
    
    ASFiNAG has seven single performance indicators:
        Griffigkeit -> Skid_Resistance
        Spurrinnen -> Transverse_Evenness (Rutting)
        Längsebenheit -> Longitudinal_Evenness
        Oberflächenschäden -> Surface_Defects
        Risse -> Cracking (Single and Aligator)
        Alter Decke -> Age of asphalt structure
        Theor. Tragfähigkeit -> Bearing_Capacity (theoretical)
    """

    @staticmethod
    def standardize_transverse_evenness(ZG_SR):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_SR (CV_R) -> ZW_SR (CI_R)
        
        Spurrinnen === Transverse_Evenness (Rutting)
        """
        ZW_SR = 1 + 0.175 * ZG_SR
        return np.clip(ZW_SR, 1, 5)
    
    @staticmethod
    def standardize_skid_resistance(ZG_GR):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_GR (CV_SR) -> ZW_GR (CI_SR)
        
        Griffigkeit === Skid_Resistance
        """
        ZW_GR = np.where(ZG_GR <= 0.45,
                         9.9286 - 14.286 * ZG_GR,
                         6.5 - 6.6667 * ZG_GR)
        return np.clip(ZW_GR, 1, 5)
        
    @staticmethod
    def standardize_longitudinal_evenness(ZG_LE):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_LE (CV_LE) -> ZW_LE (CI_LE)
        
        Längsebenheit === Longitudinal_Evenness
        """
        ZW_LE = 1 + 0.7778 * ZG_LE
        return np.clip(ZW_LE, 1, 5)
    
    @staticmethod
    def standardize_crack(ZG_RI):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_RI (CV_C) -> ZW_RI (CI_C)
        
        Risse === Cracking (Single and Aligator)
        """
        ZW_RI = 1 + 0.35 * ZG_RI
        return np.clip(ZW_RI, 1, 5)
    
    @staticmethod
    def standardize_surface_damage(ZG_OS):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_OS (CV_SD) -> ZW_OS (CI_SD)
        
        Oberflächenschäden === Surface_Defects
        """
        ZW_OS = 1 + 0.0875 * ZG_OS
        return np.clip(ZW_OS, 1, 5)
        
    @staticmethod
    def standardize_bearing_capacity(ZG_Tragf):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_Tragf -> SI_Tragf (SI_BC)
        
        Theor. Tragfähigkeit === Bearing_Capacity (theoretical)
        
        4.3.6.3 Substanzteilwert Tragfähigkeit (Substance value bearing capacity)
        
        ZG_Tragf === Theoretical load-bearing capacity
        
        Berechnung Zustandsgröße über Prognosemodell (Alter, VBI) === Calculation of state variables via forecast model (age, VBI)
        
        More info in chapter 4.2.3 and chapter 4.2.4.
        
        """
        SI_Tragf = 1 + 0.35 * ZG_Tragf
        return np.clip(SI_Tragf, 1, 5)
    
    @staticmethod
    def standardize_age_of_asphalt_structure(Alter_Decke, asphalt_surface_thickness):
        """
        Convert the age of the surface (ceiling) asphalt pavements to condition index.
        
        Alter_Decke (CV_age) -> ZW_Alter (CI_age)
        
        Alter Decke === Age of asphalt structure
        
        -----------------
        
        Manual Pavement management in Austria - Section 4.3.6.1
        
        The following approach is chosen for asphalt pavements based on 
        the average service life according to [2].
        
        Asphalt thickness must be in cm.
        Age must be in years.
        
        Asphaltdecke mit einer Gesamtdicke == Total thickness of asphalt surface
        
        [2] Weninger-Vycudil A.: Entwicklung von Systemelementen für ein österreichisches PMS. 
        Dissertation, ausgeführt am Institut für Straßenbau und Straßenerhaltung, Technische 
        Universität Wien, 2001
        
        How to calculate the age? What is the time reference?
        """
        ZW_Alter = np.where(asphalt_surface_thickness > 2,
                            0.21 * Alter_Decke - 0.17,
                            0.30 * Alter_Decke - 0.17)
        return np.clip(ZW_Alter, 1, 5)
    
    @staticmethod
    def calculate_safety_index(ZW_SR, ZW_GR):
        """
        Combine condition indexes.
        
        ZW_GR (CI_SR) + ZW_SR (CI_R) -> GI_Sicherheit (CSI_safety)
        
        Calculates the safety index based on transverse evenness and skid resistance.
        
        Gebrauchsteilwert Sicherheit === Safety index 
        """
        safety_index = np.maximum(ZW_SR, ZW_GR) + 0.1 * np.minimum(ZW_SR, ZW_GR) - 0.1
        return np.clip(safety_index, 1, 5)
    
    @staticmethod
    def calculate_comfort_index(ZW_LE, ZW_OS):
        """
        Combine condition indexes.
        
        ZW_LE (CI_LE) + ZW_OS (CI_SD) -> GI_Komfort. (CSI_Comfort)
        
        Calculates the comfort index based on longitudinal evenness and surface defects.
        
        Gebrauchsteilwert Fahrkomfort === Comfort index
        """
        
        #Convert Condition index (ZW_OS) to Conditional value (ZG_OS)
        ZG_OS = (ZW_OS - 1) / 0.0875
        ZW_OS = 1 + 0.0021875 * ZG_OS ** 2
        
        comfort_index = (np.maximum(ZW_LE, ZW_OS)
                         + 0.1 * np.minimum(ZW_LE, ZW_OS) 
                         - 0.1)
        return np.clip(comfort_index, 1, 5)
    
    @staticmethod
    def calculate_functional_index(GI_Sicherheit, GI_Komfort):
        """
        Combine condition indexes.
        
        GI_Sicherheit (CSI_safety) + GI_Komfort. (CSI_Comfort) -> GI (CSI)
        
        Calculates the functional index based on safety and comfort indexes.
        
        Gebrauchswert gesamt === Functional index         
        """
        functional_index = (np.maximum(GI_Sicherheit, GI_Komfort) 
                            + 0.1 * np.minimum(GI_Sicherheit, GI_Komfort) 
                            - 0.1)
        return np.clip(functional_index, 1, 5)
    
    @staticmethod
    def calculate_surface_structural_index(ZW_RI, ZW_OS, ZW_SR, 
                                           ZW_LE, Alter_Decke, asphalt_surface_thickness):
        """
        Calculates the surface structural index for a given pavement based on 
        its condition indicators and properties (age and asphalt thicknes).
        
        ZW_RI    (CI_C)
        ZW_OS    (CI_SD)
        ZW_SR    (CI_R)       -> SI_Decke (SI_AS)
        ZW_LE    (CI_LE)
        ZW_Alter (CI_age)
        """
        # Constants for condition value conversion
        TRANSVERSAL_EVENNESS_FACTOR = 0.175
        LONGITUDINAL_EVENNESS_FACTOR = 0.7778
        
        #Convert Condition index to Conditional value
        def convert_to_conditional_value(index, factor):
            return (index - 1) / factor
        
        ZG_SR = convert_to_conditional_value(ZW_SR, TRANSVERSAL_EVENNESS_FACTOR)
        ZG_LE = convert_to_conditional_value(ZW_LE, LONGITUDINAL_EVENNESS_FACTOR)
        
        ZW_AlterAS = ASFiNAG.standardize_age_of_asphalt_structure(Alter_Decke, asphalt_surface_thickness)
        
        # Determine if input is a numpy array to use numpy functions for max and min
        if isinstance(ZW_RI, np.ndarray):
            max_, min_ = np.maximum, np.minimum
        else:
            max_, min_ = max, min
            
        SI_Decke = np.max([
            np.max([ZW_RI, ZW_OS], axis=0) + 0.1*np.min([ZW_RI, ZW_OS], axis=0)-0.1,
            np.max([
                np.clip(1+0.00010938*ZG_SR**3, 1,5),
                np.clip(1+0.03840988*ZG_LE**3, 1,5),
            ], axis=0),
            np.clip(0.08*ZW_RI+0.61, 0, 0.85)*ZW_AlterAS
        ], axis=0)
        return np.clip(SI_Decke, 1, 5)
    
    @staticmethod
    def calculate_structural_index(SI_Decke, SI_Tragf, asphalt_surface_thickness, total_pavement_thickness):
        """
        Calculates the structural index based on surface structural index and theoretical bearing capacity.
        
        SI_Decke (SI_AS) + SI_Tragf (SI_BC) -> SI_gesamt (SI)
        
        DickeDecke === thickness of the top layer [cm]
        DickeGebSchichten === Total thickness of all bound layers [cm]
        """
        
        SI_gesamt = (
            (SI_Decke * asphalt_surface_thickness + SI_Tragf * total_pavement_thickness)
            / (asphalt_surface_thickness + total_pavement_thickness)
        )
        return np.clip(SI_gesamt, 1, 5)
        
    @staticmethod
    def calculate_global_index(GI, SI, street_category):
        """
        GI = OSI
        """
        WGI = 1
        if street_category == 'highway':
            WSI = 0.89
        if street_category == 'country_road':
            WSI = 0.8
        return np.clip(np.maximum(WGI*GI, WSI*SI), 1, 5)
    
    single_performance_index = {
        'Skid_Resistance': 'Skid_Resistance',
        'Transverse_Evenness': 'Transverse_Evenness',
        'Longitudinal_Evenness': 'Longitudinal_Evenness',
        'Surface_Defects': 'Surface_Defects',
        'Cracking': 'Cracking',
        'Bearing_Capacity': 'Bearing_Capacity',
    }
    
    combined_performance_index = {
        'Safety': ['Skid_Resistance', 'Transverse_Evenness'],
        'Comfort': ['Longitudinal_Evenness', 'Surface_Defects'],
        'Surface_Structural': ['Cracking', 'Surface_Defects', 'Transverse_Evenness', 'Longitudinal_Evenness'],
        'Functional': ['Safety', 'Comfort'],
        'Structural': ['Surface_Structural', 'Bearing_Capacity'],
        'Global': ['Functional', 'Structural'],
    }

    # Dictionary containing the configuration for each indicator
    indicator_config = {
        'Safety': (['Transverse_Evenness', 'Skid_Resistance'], []),
        'Comfort': (['Longitudinal_Evenness', 'Surface_Defects'], []),
        'Functional': (['Safety', 'Comfort'], []),
        'Surface_Structural': (['Cracking', 'Surface_Defects', 'Transverse_Evenness', 'Longitudinal_Evenness'], 
                                ['age', 'asphalt_surface_thickness']),
        'Structural': (['Surface_Structural', 'Bearing_Capacity'], ['asphalt_surface_thickness', 'total_pavement_thickness']),
        'Global': (['Functional', 'Structural'], ['street_category']),
    }
    
    def __init__(self, properties):
        super().__init__(properties)
        self.name = properties.get('name', '')
        self.asphalt_surface_thickness = properties['asphalt_surface_thickness']
        self.total_pavement_thickness = properties['total_pavement_thickness']
        self.street_category = properties['street_category']
        self.date_asphalt_surface = properties['date_asphalt_surface']
        
        today = datetime.today().strftime('%d/%m/%Y')
        self.age = self._calculate_dates_difference_in_years(self.date_asphalt_surface, today)

        self.combined_indicators_variables = [
            'Safety', 
            'Comfort', 
            'Functional', 
            {
                'Surface_Structural': 
                {
                    'Alter_Decke': self.age, 
                    'asphalt_surface_thickness': self.asphalt_surface_thickness
                }
            }, 
            {
                'Structural': 
                {
                    'asphalt_surface_thickness': self.asphalt_surface_thickness, 
                    'total_pavement_thickness': self.total_pavement_thickness
                }
            }, 
            {
                'Global': 
                {
                    'street_category': self.street_category
                }
            }
        ]
        
        # Define transformation functions for single performance indicators
        self.transformation_functions = {
            'Cracking': self.standardize_crack,
            'Surface_Defects': self.standardize_surface_damage,
            'Transverse_Evenness': self.standardize_transverse_evenness,
            'Longitudinal_Evenness': self.standardize_longitudinal_evenness,
            'Skid_Resistance': self.standardize_skid_resistance,
            'Bearing_Capacity': self.standardize_bearing_capacity,
        }
        
        # Define combination functions for combined performance indicators
        self.combination_functions = {
            'Safety': self.calculate_safety_index,
            'Comfort': self.calculate_comfort_index,
            'Functional': self.calculate_functional_index,
            'Surface_Structural': self.calculate_surface_structural_index,
            'Structural': self.calculate_structural_index,
            'Global': self.calculate_global_index,
        }

##############################################################################

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError('Age cannot be negative.')
        self._age = value
    
    def get_combined_indicators(self, 
                                indicators_prediction):
        
        
        for indicator in self.combined_indicators_variables:

            if isinstance(indicator, str):
                indicators_prediction[indicator] = self.get_combined_indicator(indicator, indicators_prediction)
            
            elif isinstance(indicator, dict):
                for indicator_name, params in indicator.items():
                    indicators_prediction[indicator_name] = self.get_combined_indicator(indicator_name, indicators_prediction, **params)
        
        return indicators_prediction
    
##############################################################################

    def _prepare_arguments(self, row, columns, properties_needed, suffix):
        arguments = []
        for column in columns:
            arguments.append(row[column + suffix] if column + suffix in row else row[column])
        if properties_needed:
            for prop in properties_needed:
                section_properties = {key: self.properties[key] for key in self.properties}
                if prop == 'age':
                    if 'Date' not in row:
                        raise ValueError('Date is not in the inspection data.')
                    arguments.append(self._calculate_dates_difference_in_years(self.date_asphalt_surface, row['Date']))
                else:
                    arguments.append(section_properties[prop])

        return arguments
    
    def _combine_indicator(
        self, 
        inspections: dict, 
        indicator: str, 
        columns: List[str], 
        properties_needed: List[str] = [], 
        suffix: str = ''
    ) -> np.ndarray:
        """
        Combines indicator values from inspections data into a single value for each indicator.
        
        Args:
            inspections (dict): A dictionary containing the inspections data.
            indicator (str): The name of the indicator to combine.
            columns (List[str]): A list of column names to use for combining the indicator values.
            properties_needed (List[str], optional): A list of property names needed for combining the indicator values. Defaults to an empty list.
            suffix (str, optional): A suffix to add to the column names. Defaults to an empty string.
        
        Returns:
            np.array: An array of combined indicator values.
        """
        function = self.combination_functions[indicator]
        if suffix:
            columns = self._add_suffix(columns, suffix)

        results = []
        
        for i in range(len(next(iter(inspections.values())))):
            row_data = {key: inspections[key][i] for key in inspections}
            arguments = self._prepare_arguments(row_data, columns, properties_needed, suffix)
            result = function(*arguments)
            results.append(result)

        return np.array(results)
    
    def combine_indicator(self, indicator, inspections, to_suffix=True):
        """
        Combines indicator values from inspections data into a single value for each indicator.
        
        Args:
            indicator (str): The name of the indicator to combine.
            df_inspections (dict): A dictionary containing the inspections data.
            to_suffix (bool, optional): If True, a '_ASFiNAG' suffix will be added to the column names. Defaults to True.
        
        Returns:
            np.array: An array of combined indicator values.
        
        Raises:
            ValueError: If the indicator is not in the indicator_config dictionary.
        """
        suffix = '_ASFiNAG' if to_suffix else ''
        
        if indicator in self.indicator_config:
            columns, properties_needed = self.indicator_config[indicator]
            return self._combine_indicator(inspections, indicator, columns, properties_needed, suffix)
        else:
            raise ValueError(f"Unknown indicator: {indicator}")
