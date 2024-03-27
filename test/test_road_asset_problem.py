import unittest
import random

import numpy as np
import pandas as pd

from ams.prediction.markov import MarkovContinous
from ams.performance.performance import Performance
from ams.optimization.multi_objective_optimization import Multi_objective_optimization

from InfraROBManagementSystem.convert.ASFiNAG import ASFiNAG
from InfraROBManagementSystem.optimization.problem import InfraROBRoadProblem


class TestASFiNAGProblemProblem(unittest.TestCase):

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
                   "cost": 10
                   },
        ]

        # Create one performance model for each indicator
        performance_models = {}
        for key, theta in thetas.items():
            markov = MarkovContinous(worst_IC=5, best_IC=1)
            markov.theta = theta
            filtered_actions = InfraROBRoadProblem.extract_indicator(key, actions)
            performance_models[key] = Performance(markov, filtered_actions)

        df_properties = pd.DataFrame({'Section_Name': ['road_1'],
                              'Asphalt_Thickness': [3],
                              'Street_Category': ['highway'],
                              'Age': ['01/01/2013'],
                              })

        organization = ASFiNAG(df_properties)
        
        self.problem = InfraROBRoadProblem(performance_models, organization, time_horizon=20)

        self.action_binary = np.array([0, 0] * 5)

        self.action_binary[2] = 5 #year
        self.action_binary[3] = 1 #action_1

        self.action_binary[4] = 7 #year
        self.action_binary[5] = 1 #action_1

        self.action_binary[6] = 10 #year
        self.action_binary[7] = 2 #action_2

    def test_decode_solution(self):
        actions = self.problem._decode_solution(np.array([0, 0] * 5))

        self.assertEqual(actions,{})

        actions = self.problem._decode_solution(self.action_binary)

        self.assertEqual(actions,
                         {
                          '10': 'action_2',
                          '5': 'action_1',
                          '7': 'action_1'})

    def test_calc_budget(self):
        total_cost = self.problem._calc_budget([self.action_binary])[0]
        self.assertAlmostEqual(total_cost, 16.0243, places=3)

    def test_get_number_of_interventions(self):
        n_actions = self.problem._get_number_of_interventions(self.action_binary)
        self.assertEqual(n_actions, 3)

    def test_calc_area_under_curve(self):
        random.seed(1)
        performance = self.problem._evaluate_performance([np.array([0, 0] * 5)])
        performance = self.problem._calc_all_indicators(performance)
        area_under_curve = self.problem._calc_area_under_curve(performance)[0]

        self.assertAlmostEqual(area_under_curve['Bearing_Capacity'], 23.7, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Cracking'], 40., delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Longitudinal_Evenness'], 36, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Skid_Resistance'], 55.5, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Transverse_Evenness'], 41.5, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Surface_Defects'], 31.4, delta=1e-5)
        
        self.assertAlmostEqual(area_under_curve['Safety'], 57.55, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Comfort'], 36.22342857142857, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Functional'], 59.07234285714286, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Surface_Structural'], 41.315224191736604, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Structural'], 32.507612095868296, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Global'], 59.07234285714286, delta=1e-5)

        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]

        self.assertAlmostEqual(area_under_curve['Bearing_Capacity'], 23.7, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Cracking'], 33.9, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Longitudinal_Evenness'], 32.3, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Skid_Resistance'], 52.5, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Transverse_Evenness'], 30, delta=1e-5)

    def test_calc_area_under_curve_with_different_initial_ICs(self):
        self.problem.initial_ICs = {'Bearing_Capacity':3,
                                    'Cracking':1,
                                    'Longitudinal_Evenness':2,
                                    'Skid_Resistance':2,
                                    'Transverse_Evenness':3,
                                    }
        random.seed(1)
        performance = self.problem._evaluate_performance([np.array([0, 0] * 5)])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]

        self.assertAlmostEqual(area_under_curve['Bearing_Capacity'], 65, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Cracking'], 40., delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Longitudinal_Evenness'],  50.9, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Skid_Resistance'], 68.2, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Transverse_Evenness'], 76.6, delta=1e-5)

        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]

        self.assertAlmostEqual(area_under_curve['Bearing_Capacity'], 65, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Cracking'], 33.9, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Longitudinal_Evenness'], 48.4, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Skid_Resistance'], 71.9, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Transverse_Evenness'], 51.8, delta=1e-5)
        
        self.problem.initial_ICs = {}
    
    def test_calc_max_indicator(self):
        random.seed(1)
        performance = self.problem._evaluate_performance([np.array([0, 0] * 5)])
        performance = self.problem._calc_all_indicators(performance)
        max_indicator = self.problem._calc_max_indicator(performance)[0]

        self.assertAlmostEqual(max_indicator['Bearing_Capacity'], 1.3, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Cracking'], 3, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Longitudinal_Evenness'], 2.3, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Skid_Resistance'], 3.8, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Transverse_Evenness'], 2.8, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Surface_Defects'], 2.3, delta=1e-5)
        
        self.assertAlmostEqual(max_indicator['Safety'], 3.98, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Comfort'], 2.348285714285714, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Functional'], 4.114828571428571, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Surface_Structural'], 3.13, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Structural'], 2.215, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Global'], 4.114828571428571, delta=1e-5)

        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        max_indicator = self.problem._calc_max_indicator([performance])[0]

        self.assertAlmostEqual(max_indicator['Bearing_Capacity'], 1.3, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Cracking'], 2.7, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Longitudinal_Evenness'], 2.2, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Skid_Resistance'], 3.4, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Transverse_Evenness'], 2.2, delta=1e-5)

    def test_calc_max_indicators(self):
        random.seed(1)
        performance = self.problem._evaluate_performance([np.array([0, 0] * 5)])[0]
        indicators_diff = self.problem._calc_max_indicators([performance])
        
        self.assertAlmostEqual(indicators_diff[0][0], -3.7, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[1][0], -2, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[2][0], -2.7, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[3][0], -1.2, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[4][0], -2.7, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[5][0], -2.2, delta=1e-5)

        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        indicators_diff = self.problem._calc_max_indicators([performance])

        self.assertAlmostEqual(indicators_diff[0][0], -3.7, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[1][0], -2.3, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[2][0], -2.8, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[3][0], -1.6, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[4][0], -2.5, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[5][0], -2.8, delta=1e-5)

    def test_evaluate(self):
        out = {}
        random.seed(1)
        self.problem._evaluate([self.action_binary], out)

        self.assertAlmostEqual(out['F'][0][0], 54.60277142857143, delta=1e-5)
        self.assertAlmostEqual(out['F'][1][0], 16.0243, places=3)  

if __name__ == '__main__':
    unittest.main()