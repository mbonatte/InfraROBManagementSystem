import unittest
import random

import numpy as np

from pymoo.config import Config
Config.warnings['not_compiled'] = False

from ams.prediction.markov import MarkovContinous
from ams.performance.performance import Performance
from ams.optimization.multi_objective_optimization import Multi_objective_optimization

from InfraROBManagementSystem.organization.ASFiNAG import ASFiNAG
from InfraROBManagementSystem.optimization.problem import InfraROBRoadProblem

class Test_ASFiNAG_optimization(unittest.TestCase):

    def setUp(self):
        # Thetas
        PI_B = [0.0186, 0.0256, 0.0113, 0.0420]
        PI_CR = [0.0736, 0.1178, 0.1777, 0.3542]
        PI_E = [0.0671, 0.0390, 0.0489, 0.0743]
        PI_F = [0.1773, 0.2108, 0.1071, 0.0765]
        PI_R = [0.1084, 0.0395, 0.0443, 0.0378]
        PI_SD = [0.1, 0.1, 0.1, 0.1] ######

        #Mapping thetas and indicators
        thetas = {'Bearing_Capacity': PI_B,
                  'Cracking':PI_CR,
                  'Longitudinal_Evenness': PI_E,
                  'Skid_Resistance': PI_F,
                  'Transverse_Evenness': PI_R,
                  'Surface_Defects': PI_SD
                  }

        # Set actions database
        actions = [{"name": 'action_1',
                    "Bearing_Capacity": {
                        "time_of_reduction": {
                                2: [5, 5, 5],
                                3: [5, 5, 5]
                        },
                       "reduction_rate":    {
                                2: [0.1, 0.1, 0.1],
                                3: [0.1, 0.1, 0.1]
                        }
                    },
                    "Cracking": {
                        "improvement": {
                                2: [1, 1, 1],
                                3: [2, 2, 2],
                                4: [3, 3, 3],
                                5: [4, 4, 4]
                        }
                    },
                    "Skid_Resistance": {
                        "improvement": {
                            2: [1, 1, 1],
                            3: [2, 2, 2],
                            4: [3, 3, 3],
                            5: [4, 4, 4]
                        }
                    },
                   "cost": 3.70
                   },
                   {"name": 'action_2',
                    "Transverse_Evenness": {
                        "improvement": {
                            2: [1, 1, 1],
                            3: [2, 2, 2],
                            4: [3, 3, 3],
                            5: [4, 4, 4]
                        }
                    },
                    "Longitudinal_Evenness": {
                        "improvement":  {
                            2:[1, 1, 1], 
                            3:[2, 2, 2], 
                            4:[2, 2, 2]
                        }
                    },
                    "Skid_Resistance": {
                        "improvement": {
                            2: [1, 1, 1],
                            3: [2, 2, 2],
                            4: [3, 3, 3],
                            5: [4, 4, 4]
                        }
                    },
                   "cost": 3
                   },
        ]

        # Create one performance model for each indicator
        performance_models = {}
        for key, theta in thetas.items():
            markov = MarkovContinous(worst_IC=5, best_IC=1)
            markov.theta = theta
            filtered_actions = InfraROBRoadProblem.extract_indicator(key, actions)
            performance_models[key] = Performance(markov, filtered_actions)

        properties = {
            'name': 'road_1',
            'asphalt_surface_thickness': 5,
            'total_pavement_thickness': 5,
            'street_category': 'highway',
            'date_asphalt_surface': '01/01/2010',
            }

        organization = ASFiNAG(properties)
        
        problem = InfraROBRoadProblem(performance_models, organization, time_horizon=20)
        

        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 20, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':3})

    def test_minimize(self):
        np.random.seed(1)
        random.seed(1)

        res = self.optimization.minimize()

        sort = np.argsort(res.F.T)[1]

        performance = res.F.T[0][sort]
        cost = res.F.T[1][sort]
        best_action = self.optimization.problem._decode_solution(res.X[sort][-1])

        action = {'13': 'action_1', '17': 'action_2', '6': 'action_2', '9': 'action_1'}
        self.assertEqual(action, best_action)
        
        self.assertAlmostEqual(performance[0], 55.6753285, places=3)
        self.assertAlmostEqual(performance[-1], 33.5234164, places=3)

        self.assertAlmostEqual(cost[0], 0, places=3)
        self.assertAlmostEqual(cost[-1], 11.993377134757244, places=5)
        
        prediction = self.optimization.problem._get_performances(best_action)
        max_global_indicator = self.optimization.problem._calc_max_global_indicator([prediction])[0]
        self.assertAlmostEqual(max_global_indicator, 2.7292857142857145, places=5)
        

    def test_budget_constrain(self):
        max_budget = 3
        self.optimization.problem.max_budget = max_budget

        np.random.seed(1)
        random.seed(1)
        res = self.optimization.minimize()        

        cost = res.F.T[1]

        self.assertTrue(max(cost) < max_budget)

    def test_global_max_indicator_constrain(self):
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':5})
        max_global_indicator = 3
        self.optimization.problem.max_global_indicator = max_global_indicator

        np.random.seed(1)
        random.seed(1)
        res = self.optimization.minimize()
        sort = np.argsort(res.F.T)[1]

        most_expensive_solution = res.X[sort][-1]
        actions_schedule = self.optimization.problem._decode_solution(most_expensive_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        max_global_indicator = self.optimization.problem._calc_max_global_indicator([performance])
        self.assertTrue(max_global_indicator <= 3)
        
        cheapest_solution = res.X[sort][0]
        actions_schedule = self.optimization.problem._decode_solution(cheapest_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        max_global_indicator = self.optimization.problem._calc_max_global_indicator([performance])
        self.assertTrue(max_global_indicator <= 3.07)
        
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':3})

    def test_single_indicators_constrain(self):
        max_indicators = {'Bearing_Capacity': 2,
                          'Cracking': 2,
                          'Longitudinal_Evenness': 2,
                          'Skid_Resistance': 2,
                          'Transverse_Evenness': 2}
        
        self.optimization.problem.single_indicators_constrain = max_indicators

        np.random.seed(1)
        random.seed(1)
        res = self.optimization.minimize()  
        
        most_expensive_solution = res.X[-1]
        actions_schedule = self.optimization.problem._decode_solution(most_expensive_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        indicators_diff = self.optimization.problem._calc_max_indicators([performance])
        self.assertTrue(max(indicators_diff)[0] <= 0.11)
    
    def test_budget_indicator_constrain(self):
        max_budget = 3.5
        max_global_indicator = 3

        self.optimization.problem.max_budget = max_budget
        self.optimization.problem.max_global_indicator = max_global_indicator

        np.random.seed(2)
        random.seed(2)
        res = self.optimization.minimize()
        
        ## Indicator
        cheapest_solution = res.X[0]
        actions_schedule = self.optimization.problem._decode_solution(cheapest_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        max_global_indicator = self.optimization.problem._calc_max_global_indicator([performance])        
        self.assertTrue(max_global_indicator <= max_global_indicator)
        
        ## Cost
        cost = res.F.T[1]
        self.assertTrue(max(cost) < max_budget)

if __name__ == '__main__':
    unittest.main()