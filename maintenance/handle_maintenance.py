import io
import numpy as np
import pandas as pd
from prediction.markov import MarkovContinous
from prediction.handle_prediction import convert_to_markov, convert_to_markov_organization, get_fitted_markov_model
from .performance import Performance
from convert.organization import Organization
    
def get_IC_through_time_maintenance(inspections, institution, maintenance_data, maintenance_scenario, worst_IC, best_IC, time_block, time_hoziron, asset_properties = None):

    response = {}
    if institution == 'Generic':
        buffer = io.StringIO(inspections)
        df  = pd.read_csv(buffer, sep=',')
        data_markov_format = convert_to_markov(df, worst_IC, best_IC, time_block)
        markov_model = MarkovContinous(worst_IC,
                                       best_IC,
                                       optimizer=True)
        
        markov_model.fit(data_markov_format['Initial'],
                         data_markov_format['Time'],
                         data_markov_format['Final'])
        performance = Performance(markov_model, maintenance_data)

        initial_IC = best_IC
        response = {}
        response['Year'] = list(range(0, time_hoziron + 1))
        response['IC'] = list(performance.get_IC_over_time(time_hoziron,
                                                           initial_IC = best_IC,
                                                           actions_schedule=maintenance_scenario,
                                                           number_of_samples=100)
                              )
                         
    if institution == 'ASFiNAG':
        buffer = io.StringIO(inspections)
        df_inspections  = pd.read_csv(buffer, sep=';')
        buffer = io.StringIO(asset_properties)
        df_properties  = pd.read_csv(buffer, sep=';')
        organization = Organization.set_organization(institution)
        variables = {
                        'inspections': df_inspections,
                        'properties': df_properties,
                        'institution': institution,
                        'organization': organization,
                        'worst_IC': worst_IC,
                        'best_IC': best_IC,
                        'time_block': time_block,
                        'time_hoziron': time_hoziron,
                        'maintenance_data': maintenance_data,
                        'actions_schedule': maintenance_scenario,
                        'results': {}
                     }
        predict(variables)
        response = variables['results']

    
    return response

def predict(variables):
    df_predict = pd.DataFrame()
    df_predict['Time'] = np.arange(variables['time_hoziron']+1)
    predict_single_performance_index(variables)
    #self.predict_combined_performance_index(df_predict)

def predict_single_performance_index(variables):
    indicators = variables['organization'].single_performance_index
    for indicator in indicators:
        try:
            print(indicator)
            indicator = f"{indicator}_{variables['institution']}"
            model = fit_predict_model(variables, indicator)
            performance = Performance(model, variables['maintenance_data'])
            variables['results'][indicator] = {}
            variables['results'][indicator]['Time'] = [i for i in range(variables['time_hoziron']+1)]
            variables['results'][indicator]['IC'] = list(performance.get_IC_over_time(variables['time_hoziron'],
                                                                                      initial_IC = variables['best_IC'],
                                                                                      actions_schedule=variables['actions_schedule'],
                                                                                      number_of_samples=100
                                                                                      )
                                                        )
        except KeyError as e:
            print('Error: ',e)

def fit_predict_model(variables, indicator):
    organization = variables['organization']
    df_properties = variables['properties']
    df_inspections = variables['inspections']
    df_standardized = organization(df_properties).transform_performace_indicators(df_inspections)
    df = convert_to_markov_organization(df_standardized[['Section_Name','Date', indicator]],
                                        worst_IC=variables['worst_IC'],
                                        best_IC=variables['best_IC'])
                                        
    return get_fitted_markov_model(df, variables['worst_IC'], variables['best_IC'])
