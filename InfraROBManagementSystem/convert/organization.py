import logging
from abc import ABC, abstractmethod 
import numpy as np
import pandas as pd


class Organization(ABC):
    """
    A class to represent a Organization to standardize the Peformance Indica-
    tors (PI).
    
    ...

    Attributes
    ----------
    asphalt_thickness : float
        Thickness of the asphalt
    street_category : str
        Category of street, e.g. 'Highway'

    Methods
    -------
    None.
    """
    
    @staticmethod
    def set_organization(organization):
        from . import ASFiNAG
        from . import COST_354
        list_of_organizations = {'ASFiNAG': ASFiNAG.ASFiNAG,
                                 'COST_354': COST_354.COST_354}
        return list_of_organizations[organization]
        
    
    def __init__(self, properties: pd.DataFrame):
        self.properties = properties#.to_dict('records')[0]
    
    @property
    @abstractmethod
    def single_performance_index(self):
        pass
    
    @property
    @abstractmethod
    def combined_performance_index(self):
        pass
    
    
    def calculate_PI_from_TC(self, indicator, value):
        return self.transformation_functions[indicator](value)
        
    def get_combined_indicator(self, combined_indicator, all_indicators, **kwargs):
        indicators = [all_indicators[indicator] for indicator in self.combined_performance_index[combined_indicator]]
        return self.combination_functions[combined_indicator](*indicators, **kwargs)
    
    def combine_peformance_indicators(self,indicator,df_inspections,to_suffix=True):
        logging.debug(f'Combining | {indicator}')
        #self.combination_functions[indicator](df_inspections)
        return self.combine_indicator(indicator,df_inspections,to_suffix)
        
    
    def transform_performace_indicator(self,
                                         indicator: str, 
                                         df_inspections: pd.DataFrame):
        logging.debug(f'{self.__class__.__name__} | {indicator}')
        indicators = {**self.single_performance_index, 
                      **self.combined_performance_index}
        if type(indicators[indicator]) == str:
            inspections_values = df_inspections[indicators[indicator]].astype(float)
            indicator_values = self.calculate_PI_from_TC(indicator, inspections_values)
            #return indicator_values
            return self.standardize_values(indicator_values).astype(int)
        else:
            indicator_values = self.combine_peformance_indicators(indicator,df_inspections)
            #return indicator_values
            return self.standardize_values(indicator_values).astype(int)
        
    def transform_performace_indicators(self, df_inspections: pd.DataFrame):
        logging.debug(f'{self.__class__.__name__} | Performance Indicators')
        indicators = {**self.single_performance_index, 
                      **self.combined_performance_index
                      }
        for indicator in indicators:
            try:
                indicator_transformed = self.transform_performace_indicator(indicator,
                                                                            df_inspections)
                indicator_transformed = pd.Series(indicator_transformed)
                indicator_transformed.index = df_inspections.index
                column_name = f'{indicator}_{self.__class__.__name__}'
                df_inspections[column_name] = indicator_transformed
            except ValueError as e:
                logging.warning(e)
        return df_inspections
            
            
            
    def standardize_values(self, indicator_values):
        conditions = [indicator_values < 1.5,
                      ((1.5 <= indicator_values) & (indicator_values < 2.5)),
                      ((2.5 <= indicator_values) & (indicator_values < 3.5)),
                      ((3.5 <= indicator_values) & (indicator_values < 4.5)),
                      4.5 <= indicator_values,
                      ]
        values = [1, 2, 3, 4, 5]
        indicator = np.select(conditions, values)
        return indicator
