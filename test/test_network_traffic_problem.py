import unittest
import random
import numpy as np

from InfraROBManagementSystem.optimization.problem import NetworkTrafficProblem

class TestNetworkTrafficProblem(unittest.TestCase):

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
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                "Corrective_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 1000000,
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
                    "Cost": 0.8
                    },
                "Corrective_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 40,
                    "Cost": 0.8
                    },
                },
            "Road_2_2": {
                "Preventive_10": {
                    "Fuel": 10,
                    "Cost": 1
                    },
                "Preventive_2": {
                    "Fuel": 20,
                    "Cost": 0.8
                    },
                "Corrective_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 40,
                    "Cost": 0.8
                    },
                },
            "Road_3": {
                "Preventive_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Preventive_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                "Corrective_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                },
            "Road_4": {
                "Preventive_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Preventive_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                "Corrective_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Corrective_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                },
        }
        
        self.problem = NetworkTrafficProblem(section_optimization, TMS_output, actions)
        
    
    def test_calc_network_budget(self):
        population = [1, 1, 1, 1, 1]
        network_cost = self.problem._calc_network_budget(population)
        self.assertEqual(200, network_cost)
    
        population = [1, 4, 4, 1, 1]
        network_cost = self.problem._calc_network_budget(population)
        self.assertEqual(184, network_cost)
        
        population = [4, 4, 4, 4, 4]
        fuel = self.problem._calc_network_budget(population)
        self.assertEqual(160, fuel)
    
    def test_calc_fuel(self):
        population = [1, 1, 1, 1, 1]
        fuel = self.problem._calc_fuel(population)
        self.assertEqual(270, fuel)
        
        population = [1, 4, 4, 1, 1]
        fuel = self.problem._calc_fuel(population)
        self.assertEqual(330, fuel)
        
        population = [4, 4, 4, 4, 4]
        fuel = self.problem._calc_fuel(population)
        self.assertEqual(6000120, fuel)
    
    
    def test_evaluate(self):
        population = [1, 1, 1, 1, 1]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][1][0], 200)
        self.assertEqual(out['F'][2][0], 270)
        
        #########################################
        
        population = [1, 4, 4, 1, 1]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][1][0], 184)
        self.assertEqual(out['F'][2][0], 330)
        
        #########################################
        
        population = [4, 4, 4, 4, 4]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][1][0], 160)
        self.assertEqual(out['F'][2][0], 6000120)
    
if __name__ == '__main__':
    unittest.main()