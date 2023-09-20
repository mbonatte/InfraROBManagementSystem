import io
import numpy as np
import pandas as pd

import sys
import os

# Add the parent directory of 'convert' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from convert.organization import Organization


def get_converted_IC(inspections, properties, organization_name):
    buffer = io.StringIO(inspections)
    df_inspections  = pd.read_csv(buffer, sep=';')
    buffer = io.StringIO(properties)
    df_properties  = pd.read_csv(buffer, sep=';')

    
    organization = Organization.set_organization(organization_name)
    organization(df_properties).transform_performace_indicators(df_inspections)
    response = df_inspections.to_dict('records')
    
    return response

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    PARENT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    df_inspections  = pd.read_csv(PARENT_FOLDER + '\database\\road_sections_inpections.csv', sep=',')
    df_properties  = pd.read_csv(PARENT_FOLDER + '\database\\road_sections_properties.csv', sep=',')
    
    organization_name = 'ASFiNAG'
    organization = Organization.set_organization(organization_name)
    organization(df_properties).transform_performace_indicators(df_inspections)
    
    organization_name = 'COST_354'
    organization = Organization.set_organization(organization_name)
    organization(df_properties).transform_performace_indicators(df_inspections)
    
    df_inspections.to_csv(PARENT_FOLDER + '\database\\road_sections_inpections.csv', index=False)
