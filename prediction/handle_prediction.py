import io
import numpy as np
import pandas as pd
from .markov import MarkovContinous
from convert.organization import Organization

def convert_to_markov(df, worst_IC, best_IC, time_block):
    df['Data'] = pd.to_datetime(df['Data'], format='%d-%m-%Y')
    column = df.keys()
    df_markov = pd.DataFrame(columns=['Initial','Time','Final'])
    road_sections = df[column[0]].unique()
    for road_section in road_sections:
        section = df[df[column[0]]==road_section]
        section = section.sort_values(by=column[1])
        for i in range(len(section)-1):
            if worst_IC > best_IC:
                if section.iloc[i][2] <= section.iloc[i+1][2]:
                    time_between = (section.iloc[i+1][1] -  section.iloc[i][1]).total_seconds()
                    if time_block == 'month':
                        time_between = round(time_between*3.8052e-7)
                    if time_block == 'year':
                        time_between = round(time_between*3.171e-8)
                    df_markov.loc[len(df_markov)] = [section.iloc[i][2], time_between, section.iloc[i+1][2]]
    return df_markov.astype('int', errors = 'ignore')
    

def convert_to_markov_organization(df, worst_IC, best_IC):
    column = df.keys()
    df_markov = pd.DataFrame(columns=['Initial','Time','Final'])
    road_sections = df[column[0]].unique()
    for road_section in road_sections:
        section = df[df[column[0]]==road_section]
        section = section.sort_values(by=column[1])
        for i in range(len(section)-1):
            if worst_IC > best_IC:
                if section.iloc[i][2] <= section.iloc[i+1][2]:
                    time_between = section.iloc[i+1][1] -  section.iloc[i][1]
                    df_markov.loc[len(df_markov)] = [section.iloc[i][2], time_between, section.iloc[i+1][2]]
    return df_markov.astype('int', errors = 'ignore')


def get_IC_through_time(csv_data, institution, worst_IC, best_IC, time_block, time_hoziron, asset_properties = None):
    response = {}
    if institution == 'Generic':
        buffer = io.StringIO(csv_data)
        df  = pd.read_csv(buffer, sep=',')
        data_markov_format = convert_to_markov(df, worst_IC, best_IC, time_block)
        markov_model = MarkovContinous(worst_IC,
                                       best_IC,
                                       optimizer=True)
        
        markov_model.fit(data_markov_format['Initial'],
                         data_markov_format['Time'],
                         data_markov_format['Final'])

        initial_IC = best_IC
        response['Year'] = list(range(0, time_hoziron + 1))
        response['IC'] = list(markov_model.get_mean_over_time(time_hoziron,
                                                           initial_IC))
    if institution == 'ASFiNAG':
        buffer = io.StringIO(csv_data)
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
            indicator = f"{indicator}_{variables['institution']}"
            model = fit_predict_model(variables, indicator)
            variables['results'][indicator] = {}
            variables['results'][indicator]['Time'] = [i for i in range(variables['time_hoziron']+1)]
            variables['results'][indicator]['IC'] = list(model.get_mean_over_time(variables['time_hoziron']))
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
    model = MarkovContinous(worst_IC = variables['worst_IC'],
                            best_IC = variables['best_IC'])

    model.fit(df['Initial'],df['Time'],df['Final'])
    return model
