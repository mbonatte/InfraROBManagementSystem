import io
import numpy as np
import pandas as pd
from prediction.markov import MarkovContinous
from .performance import Performance

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
    

def get_IC_through_time_maintenance(inspections, maintenance_data, maintenance_scenario, worst_IC, best_IC, time_block, time_hoziron):
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

    
    return response
