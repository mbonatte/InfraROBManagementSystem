import io
import json

import numpy as np
import pandas as pd

from ams.prediction.markov import MarkovContinous
from ams.performance.performance import Performance
from ams.optimization.problem import MaintenanceSchedulingProblem
from ams.optimization.multi_objective_optimization import Multi_objective_optimization

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

def get_markov_model(inspections, worst_IC, best_IC, time_block):
    buffer = io.StringIO(inspections)
    df  = pd.read_csv(buffer, sep=',')
    data_markov_format = convert_to_markov(df, worst_IC, best_IC, time_block)
    markov_model = MarkovContinous(worst_IC,
                                   best_IC,
                                   optimizer=True)
    
    markov_model.fit(data_markov_format['Initial'],
                     data_markov_format['Time'],
                     data_markov_format['Final'])
    return markov_model

def get_IC_through_time_maintenance(inspections, maintenance_data, maintenance_scenario, worst_IC, best_IC, time_block, time_horizon):
    
    markov_model = get_markov_model(inspections, worst_IC, best_IC, time_block, time_horizon)
    performance = Performance(markov_model, maintenance_data)

    initial_IC = best_IC
    response = {}
    response['Year'] = list(range(0, time_horizon + 1))
    response['IC'] = list(performance.get_IC_over_time(time_horizon,
                                                       initial_IC = best_IC,
                                                       actions_schedule=maintenance_scenario,
                                                       number_of_samples=100)
                          )
    return response

def get_pareto_curve(inspections, maintenance_data, worst_IC, best_IC, time_block, time_horizon):
    markov_model = get_markov_model(inspections, worst_IC, best_IC, time_block)
    problem = MaintenanceSchedulingProblem(markov_model, maintenance_data, time_horizon)

    optimization = Multi_objective_optimization()
    optimization.set_problem(problem)

    res = optimization.minimize()

    X = res.X               # Design space values
    F = res.F               # Objective spaces
    opt = res.opt           # The solutions as a Population object.
    pop = res.pop           # The final Population
    history = res.history   # The history of the algorithm. (only if save_history has been enabled during the algorithm initialization)


    #print("# Design space values\n", X)
    #print(problem.decode_binary_to_dict(X))
    #print("# Objective spaces\n", F)
    #print('# The solutions as a Population object (X)\n', opt.get("X"))
    #print('# The solutions as a Population object (F)\n', opt.get("F"))
    #print('# The final Population (X)\n = ', pop.get("X"))
    #print('# The final Population (F)\n = ', pop.get("F"))
    #print(history)
 
    F = np.array(F).T
    sort = np.argsort(F)[1]
    
    response = {}
    
    response['Performance'] = list(F[0][sort])
    response['Cost'] = list(F[1][sort])
    response['Actions_schedule'] = [problem._decode_solution(item) for item in X[sort]]
    
    response['Markov'] = []
    performance_model = Performance(markov_model, maintenance_data)
    for actions_schedule in response['Actions_schedule']:
        result = {}
        result['Time'] = list(range(0, time_horizon + 1))
        result['IC'] = list(performance_model.get_IC_over_time(time_horizon=time_horizon,
                                                                actions_schedule=actions_schedule
                                                                )
                            )
        response['Markov'].append(result)
    
    response['Dummies'] = {}
    response['Dummies']['Performance'] = list(np.array(pop.get("F")).T[0])
    response['Dummies']['Cost'] = list(np.array(pop.get("F")).T[1])
    return response

def get_pareto_curve_all_roads():
    from pathlib import Path
    MAIN_FOLDER = Path(__file__).parent.parent.resolve()
    
    path_properties = MAIN_FOLDER / './database/road_sections_properties.csv'
    road_properties = pd.read_csv(path_properties).to_dict('records')
    
    path = MAIN_FOLDER / 'database/ActionsEffects.json'
    with open(path, "r") as file:
        maintenance_data = json.load(file)
    
    time_horizon = 50
    
    markov_model = MarkovContinous(5, 1)
    markov_model.theta = [0.0736, 0.1178, 0.1777, 0.3542]
    problem = MaintenanceSchedulingProblem(markov_model, maintenance_data, time_horizon)

    optimization = Multi_objective_optimization()
    optimization.set_problem(problem)
    
    optimization_outputs = {}
    
    for road in road_properties:
        res = optimization.minimize()
        optimization_output = get_optimization_output(optimization, res)
        optimization_outputs[road['Section_Name']] = optimization_output
    
    # Writing to optimization_output.json
    path = MAIN_FOLDER / 'database/optimization_output_.json'
    with open(path, "r+") as file:
        data = json.load(file)
        data.update(optimization_outputs)
        file.seek(0)
        file.write(json.dumps(data,
                              indent=4,
                              cls=NumpyEncoder))
    
    return {}
    # return optimization_output
    
def get_optimization_output(optimization, res):
    problem = optimization.problem
    performance_model = problem.performance_model
    
    time_horizon = 50
    
    X = res.X               # Design space values
    F = res.F               # Objective spaces
    opt = res.opt           # The solutions as a Population object.
    pop = res.pop           # The final Population
    history = res.history   # The history of the algorithm. (only if save_history has been enabled during the algorithm initialization)
 
    F = np.array(F).T
    sort = np.argsort(F)[0] # Sort the performance
    
    optimization_output = {}
    
    optimization_output['Performance'] = np.round(F[0][sort], 2)
    optimization_output['Cost'] = np.round(F[1][sort], 2)
    optimization_output['Actions_schedule'] = [problem._decode_solution(item) for item in X[sort]]
    
    optimization_output['Markov'] = []
    
    for actions_schedule in optimization_output['Actions_schedule']:
        result = {}
        result['Time'] = list(range(0, time_horizon + 1))
        result['IC'] = list(performance_model.get_IC_over_time(time_horizon=time_horizon,
                                                                actions_schedule=actions_schedule
                                                                )
                            )
        optimization_output['Markov'].append(result)
    
    optimization_output['Dummies'] = {}
    optimization_output['Dummies']['Performance'] = np.round(np.array(pop.get("F")).T[0], 2)
    optimization_output['Dummies']['Cost'] = np.round(np.array(pop.get("F")).T[1], 2)
    
    return optimization_output