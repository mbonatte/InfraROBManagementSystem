from .organization import Organization
import numpy as np

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
                                
    
    
    def __init__(self, properties):
        super().__init__(properties)
        self.asphalt_surface_thickness = properties.get('Asphalt_Thickness', 0)
        self.street_category = properties.get('Street_Category', '')
        self.street_age = properties.get('Age', 0)
        
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
    
    def convert_indicator(self, indicator):
        return self.standardize_function[indicator]

##############################################################################

    def standardize_transverse_evenness(self, ZG_SR):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_SR (CV_R) -> ZW_SR (CI_R)
        
        Spurrinnen === Transverse_Evenness (Rutting)
        """
        ZW_SR = 1 + 0.175 * ZG_SR
        return np.clip(ZW_SR, 1, 5)
    
    def standardize_skid_resistance(self, ZG_GR):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_GR (CV_SR) -> ZW_GR (CI_SR)
        
        Griffigkeit === Skid_Resistance
        """
        ZW_GR = np.where(ZG_GR <= 0.45,
                         9.9286 - 14.286 * ZG_GR,
                         6.5 - 6.6667 * ZG_GR)
        return np.clip(ZW_GR, 1, 5)
        
    def standardize_longitudinal_evenness(self, ZG_LE):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_LE (CV_LE) -> ZW_LE (CI_LE)
        
        Längsebenheit === Longitudinal_Evenness
        """
        ZW_LE = 1 + 0.7778 * ZG_LE
        return np.clip(ZW_LE, 1, 5)
    
    def standardize_crack(self, ZG_RI):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_RI (CV_C) -> ZW_RI (CI_C)
        
        Risse === Cracking (Single and Aligator)
        """
        ZW_RI = 1 + 0.35 * ZG_RI
        return np.clip(ZW_RI, 1, 5)
    
    def standardize_surface_damage(self, ZG_OS):
        """
        Convert the tecnical parameter to condition index.
        
        ZG_OS (CV_SD) -> ZW_OS (CI_SD)
        
        Oberflächenschäden === Surface_Defects
        """
        ZW_OS = 1 + 0.0875 * ZG_OS
        return np.clip(ZW_OS, 1, 5)
        
    def standardize_bearing_capacity(self, ZG_Tragf):
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
    
    def standardize_age_of_asphalt_structure(self, Alter_Decke, asphalt_surface_thickness):
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
        
##############################################################################
    
    def calculate_safety_index(self, ZW_SR, ZW_GR):
        """
        Combine condition indexes.
        
        ZW_GR (CI_SR) + ZW_SR (CI_R) -> GI_Sicherheit (CSI_safety)
        
        Calculates the safety index based on transverse evenness and skid resistance.
        
        Gebrauchsteilwert Sicherheit === Safety index 
        """
        safety_index = np.maximum(ZW_SR, ZW_GR) + 0.1 * np.minimum(ZW_SR, ZW_GR) - 0.1
        return np.clip(safety_index, 1, 5)
    
    def calculate_comfort_index(self, ZW_LE, ZW_OS):
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
    
    def calculate_functional_index(self, GI_Sicherheit, GI_Komfort):
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
       
    def calculate_surface_structural_index(self, ZW_RI, ZW_OS, ZW_SR, 
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
        
        ZW_AlterAS = self.standardize_age_of_asphalt_structure(Alter_Decke, asphalt_surface_thickness)
        
        # Determine if input is a numpy array to use numpy functions for max and min
        if isinstance(ZW_RI, np.ndarray):
            max_, min_ = np.maximum, np.minimum
        else:
            max_, min_ = max, min
            
        SI_Decke = max_(
            max_(ZW_RI, ZW_OS) + 0.1*min_(ZW_RI, ZW_OS)-0.1,
            max_(
                min_(1+0.00010938*ZG_SR**3, 5),
                min_(1+0.03840988*ZG_LE**3, 5)
            ),
            min_(0.08*ZW_RI+0.61, 0.85)*ZW_AlterAS
        )
        
        return np.clip(SI_Decke, 1, 5)
    
    def calculate_structural_index(self, SI_Decke, SI_Tragf, asphalt_surface_thickness, total_pavement_thickness):
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
        
    def calculate_global_index(self, GI, SI, street_category):
        """
        GI = OSI
        """
        WGI = 1
        if street_category == 'highway':
            WSI = 0.89
        if street_category == 'country_road':
            WSI = 0.8
        return np.clip(np.maximum(WGI*GI, WSI*SI), 1, 5)
    
##############################################################################
    
    def get_conbined_indicators(self, 
                                indicators_prediction, 
                                age, 
                                asphalt_surface_thickness, 
                                total_pavement_thickness,
                                street_category):
        for indicator in ['Safety', 'Comfort', 'Functional', 
                          {'Surface_Structural': {'Alter_Decke': age, 'asphalt_surface_thickness': asphalt_surface_thickness}}, 
                          {'Structural': {'asphalt_surface_thickness': asphalt_surface_thickness, 'total_pavement_thickness': total_pavement_thickness}}, 
                          {'Global': {'street_category': street_category}}]:

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
            section_properties = self.properties.loc[self.properties['Section_Name'] == row['Section_Name']].iloc[0]
            for prop in properties_needed:
                if prop == 'Age' and 'Date' in row:
                    arguments.append(self._calculate_dates_difference_in_years(section_properties[prop], row['Date']))
                else:
                    arguments.append(section_properties[prop])

        return arguments

    def _combine_indicator(self, df_inspections, indicator, columns, properties_needed=[], suffix=''):
        function = self.combination_functions[indicator]
        if suffix:
            columns = self._add_suffix(columns, suffix)

        results = []
        for _, row in df_inspections.iterrows():
            arguments = self._prepare_arguments(row, columns, properties_needed, suffix)
            result = function(*arguments)
            results.append(result)

        return np.array(results)
    
    def combine_indicator(self, indicator,df_inspections,to_suffix=True):
        suffix = '_ASFiNAG' if to_suffix else ''
        
        indicator_config = {
            'Safety': (['Transverse_Evenness', 'Skid_Resistance'], []),
            'Comfort': (['Longitudinal_Evenness', 'Surface_Defects'], []),
            'Functional': (['Safety', 'Comfort'], []),
            'Surface_Structural': (['Cracking', 'Surface_Defects', 'Transverse_Evenness', 'Longitudinal_Evenness'], ['Age', 'Asphalt_Thickness']),
            'Structural': (['Surface_Structural', 'Bearing_Capacity'], ['Asphalt_Thickness', 'Total_Pavement_Thickness']),
            'Global': (['Functional', 'Structural'], ['Street_Category']),
        }
        
        if indicator in indicator_config:
            columns, properties_needed = indicator_config[indicator]
            return self._combine_indicator(df_inspections, indicator, columns, properties_needed, suffix)
        else:
            raise ValueError(f"Unknown indicator: {indicator}")

##############################################################################
            
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