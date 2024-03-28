import logging
from abc import ABC, abstractmethod 

from datetime import datetime

import numpy as np
import pandas as pd


class Organization(ABC):
    """
    Abstract class to represent an organization for standardizing Performance Indicators (PI).

    This class provides the framework for transforming and combining performance indicators
    according to the organization-specific methodologies.

    Attributes:
        properties (pd.DataFrame): Dataframe containing the properties relevant to the organization.
    
    Methods:
        calculate_PI_from_TC(indicator, value): Abstract method for calculating performance indicators from test conditions.
        get_combined_indicator(combined_indicator, all_indicators, **kwargs): Abstract method for getting combined indicators.
    """
    
    @staticmethod
    def set_organization(organization):
        from . import ASFiNAG
        from . import COST_354
        list_of_organizations = {'ASFiNAG': ASFiNAG.ASFiNAG,
                                 'COST_354': COST_354.COST_354}
        return list_of_organizations[organization]
        
    def __init__(self, properties: pd.DataFrame):
        """
        Initializes the Organization with the given properties.

        Parameters:
            properties (pd.DataFrame): A dataframe containing properties relevant to the organization.
        """
        self.properties = properties
    
    @property
    @abstractmethod
    def single_performance_index(self):
        """Abstract property to define single performance indicators."""
        pass
    
    @property
    @abstractmethod
    def combined_performance_index(self):
        """Abstract property to define how single performance indicators are combined."""
        pass
    
    def _add_suffix(self, columns, suffix):
        return [f"{column}{suffix}" for column in columns]
    
    def _calculate_dates_difference_in_years(self, start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
        end_date = datetime.strptime(end_date_str, "%d/%m/%Y")
        return round((end_date - start_date).days / 365)
    
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
