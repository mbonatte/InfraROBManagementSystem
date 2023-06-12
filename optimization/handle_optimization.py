import io
import numpy as np
import pandas as pd
from prediction.markov import MarkovContinous
from maintenance.performance import Performance
from .problem import MyProblem
from .multi_objective_optimization import Multi_objective_optimization

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

def get_IC_through_time_maintenance(inspections, maintenance_data, maintenance_scenario, worst_IC, best_IC, time_block, time_hoziron):
    
    markov_model = get_markov_model(inspections, worst_IC, best_IC, time_block, time_hoziron)
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


def get_pareto_curve(inspections, maintenance_data, worst_IC, best_IC, time_block, time_hoziron):
    markov_model = get_markov_model(inspections, worst_IC, best_IC, time_block)
    problem = MyProblem(markov_model, maintenance_data, time_hoziron)

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
    response['Actions_schedule'] = [problem.decode_binary_to_dict(item) for item in X[sort]]
    
    response['Markov'] = []
    performance_model = Performance(markov_model, maintenance_data)
    for actions_schedule in response['Actions_schedule']:
        result = {}
        result['Time'] = list(range(0, time_hoziron + 1))
        result['IC'] = list(performance_model.get_IC_over_time(time_horizon=time_hoziron,
                                                                actions_schedule=actions_schedule
                                                                )
                            )
        response['Markov'].append(result)
    
    response['Dummies'] = {}
    response['Dummies']['Performance'] = list(np.array(pop.get("F")).T[0])
    response['Dummies']['Cost'] = list(np.array(pop.get("F")).T[1])
    print(response)
    return response

    #plot_pareto_curve(F[0][sort], F[1][sort])
    #X = X[sort]
    #actions = list(map(problem.decode_binary_to_dict, X))

    #plot_performances(problem.performance_model, actions)
    #plot_cost(problem, actions)

