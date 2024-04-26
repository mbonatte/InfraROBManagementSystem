import logging
from abc import ABC, abstractmethod 

from datetime import datetime

import numpy as np

class Organization(ABC):
    """
    Abstract class to represent an organization for standardizing Performance Indicators (PI).

    This class provides the framework for transforming and combining performance indicators
    according to the organization-specific methodologies.

    Attributes:
        properties : The properties relevant to the organization.
    """
    
    @staticmethod
    def set_organization(organization):
        """Return an instance of the specified organization class."""
        from . import ASFiNAG
        from . import COST_354
        
        organizations = {
            'ASFiNAG': ASFiNAG.ASFiNAG,
            'COST_354': COST_354.COST_354
        }
        return organizations[organization]
        
    def __init__(self, properties):
        """
        Initializes the Organization with the given properties.

        Parameters:
            properties (pd.DataFrame): Dataframe containing the properties relevant to the organization.
        """
        self.properties = properties
    
    @property
    @abstractmethod
    def worst_IC(self):
        """Abstract property to define the worst Index Condition."""
        pass
    
    @property
    @abstractmethod
    def best_IC(self):
        """Abstract property to define the best Index Condition."""
        pass

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
        """Add a suffix to a list of column names."""
        return [f"{column}{suffix}" for column in columns]
    
    def _calculate_dates_difference_in_years(self, start_date_str, end_date_str):
        """Calculate the difference between two dates in years."""
        start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
        end_date = datetime.strptime(end_date_str, "%d/%m/%Y")
        years = round((end_date - start_date).days / 365)
        if years < 0:
            raise ValueError('Years cannot be negative.')
        return years
    
    def calculate_PI_from_TC(self, indicator, value):
        return self.transformation_functions[indicator](value)
        
    def get_combined_indicator(self, combined_indicator, all_indicators, **kwargs):
        """Get a combined performance indicator."""
        indicators = [all_indicators[indicator] for indicator in self.combined_performance_index[combined_indicator]]
        return self.combination_functions[combined_indicator](*indicators, **kwargs)
    
    def combine_peformance_indicators(self,indicator,df_inspections,to_suffix=True):
        """Combine performance indicators."""
        logging.debug(f'Combining | {indicator}')
        return self.combine_indicator(indicator,df_inspections,to_suffix)
        
    
    def transform_performance_indicator(self,
                                         indicator: str, 
                                         inspections: list):
        logging.debug(f'{self.__class__.__name__} | {indicator}')
        indicators = {**self.single_performance_index, 
                      **self.combined_performance_index}
        if type(indicators[indicator]) == str:
            inspections_values = np.array(inspections[indicators[indicator]])
            indicator_values = self.calculate_PI_from_TC(indicator, inspections_values)
            return self.standardize_values(indicator_values).astype(int)
        else:
            indicator_values = self.combine_peformance_indicators(indicator,inspections)
            return self.standardize_values(indicator_values).astype(int)
        
    def transform_performance_indicators(self, inspections):
        logging.debug(f'{self.__class__.__name__} | Performance Indicators')
        indicators = {**self.single_performance_index, 
                      **self.combined_performance_index
                      }
        for indicator in indicators:
            try:
                indicator_transformed = self.transform_performance_indicator(indicator,
                                                                            inspections)
                column_name = f'{indicator}_{self.__class__.__name__}'
                inspections[column_name] = indicator_transformed
            except ValueError as e:
                logging.warning(e)
        
        return inspections
    
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
