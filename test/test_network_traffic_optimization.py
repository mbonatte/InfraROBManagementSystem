import unittest
import random
import numpy as np

from ams.optimization.multi_objective_optimization import Multi_objective_optimization

from InfraROBManagementSystem.optimization.problem import NetworkTrafficProblem
   
class TestNetworkTrafficOptimization(unittest.TestCase):
    def setUp(self):
        actions = [
            {
            "name": "Preventive",
            "cost": 10
            },
            {
            "name": "Corrective",
            "cost": 30
            },
        ]
        
        section_optimization = {
            "Road_1": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
            "Road_2_1": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
            "Road_2_2": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
            "Road_3": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
            "Road_4": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
        }
        
        TMS_output = {
            "Road_1": {
                "Preventive_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Preventive_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                "Corrective_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                },
            "Road_2_1": {
                "Preventive_10": {
                    "Fuel": 10,
                    "Cost": 1
                    },
                "Preventive_2": {
                    "Fuel": 20,
                    "Cost": 0.3
                    },
                "Corrective_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 40,
                    "Cost": 0.3
                    },
                },
            "Road_2_2": {
                "Preventive_10": {
                    "Fuel": 10,
                    "Cost": 1
                    },
                "Preventive_2": {
                    "Fuel": 20,
                    "Cost": 0.3
                    },
                "Corrective_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 40,
                    "Cost": 0.3
                    },
                },
            "Road_3": {
                "Preventive_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Preventive_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                "Corrective_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                },
            "Road_4": {
                "Preventive_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Preventive_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                "Corrective_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                },
        }
        
        problem = NetworkTrafficProblem(section_optimization, TMS_output, actions)
        
        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 30, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':200})
        
    def test_minimize(self):
        np.random.seed(1)
        
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        filter_fuel = res.F.T[2] < 10000000
        
        performance = res.F.T[0][filter_fuel]
        cost = res.F.T[1][filter_fuel]
        fuel = res.F.T[2][filter_fuel]
        
        sort_fuel = np.argsort(fuel)
        
        # self.assertEqual(performance[0], 100)
        # self.assertEqual(performance[-1], 20)
        
        # self.assertEqual(cost[0], 20)
        # self.assertEqual(cost[-1], 100)    


if __name__ == '__main__':
    unittest.main()