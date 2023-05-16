import io
import numpy as np
import pandas as pd
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
