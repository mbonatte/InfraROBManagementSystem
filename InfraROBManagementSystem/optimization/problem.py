"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

from pymoo.core.problem import Problem

from ams.optimization.problem import MultiIndicatorProblem, NetworkProblem

import numpy as np

class InfraROBRoadProblem(MultiIndicatorProblem):
    """
    Maintenance scheduling optimization problem.
    """
    
    def __init__(self, performance_models, organization, time_horizon, **kwargs):
        """
        Initialize the problem.

        Args:
            time_horizon: Planning horizon
        """
        super().__init__(performance_models, time_horizon, **kwargs)
        self.organization = organization
        self.age = 10
        self.asphalt_thickness = 5
        self.street_category = 'highway'
    
    def _calc_all_indicators(self, performances):
        for indicators_prediction in performances:
            indicators_prediction = self.organization.get_conbined_indicators(indicators_prediction, self.age, self.asphalt_thickness, self.street_category)
        return performances
        
    def _calc_global_area_under_curve(self, performances):
        performances = self._calc_all_indicators(performances)
        results = self._calc_area_under_curve(performances)
        return np.array([result['Global'] for result in results])
        
    def _calc_max_global_indicator(self, performances):
        performances = self._calc_all_indicators(performances)
        results = self._calc_max_indicator(performances)
        return np.array([result['Global'] for result in results])

        
class NetworkTrafficProblem(NetworkProblem):
    """
    Network optimization problem considering the traffic impact.
    """

    def __init__(self, section_optimization, TMS_output, actions, **kwargs):
        """
        Initialize the network traffic problem.

        Args:
            section_optimization: dictionary with results from the section optimization
        """
        self.section_optimization = section_optimization
        self.TMS_output = TMS_output
        self.actions = actions
        
        # Constrains
        self.max_fuel = np.inf
        
        n_sections = len(self.section_optimization) # Number of sections in the analysis
        
        xl = [0] * n_sections
        
        # Multiply by 2 to consider the maintenance approach (full and partial closure)
        xu = [len(n['Performance'])*2-1 for n in self.section_optimization.values()]

        Problem.__init__(
            self,
            n_var=n_sections,
            n_obj=3,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
            vtype=int,
            **kwargs
            )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objective functions.
        """
        
        # Objectives
        
        # Minimize network performance indicator
        f1 = self._calc_network_performance_indicator_pop(x)

        # Minimize cost
        f2 = self._calc_network_budget_pop(x)
        
        # Minimize fuel consuption
        f3 = self._calc_fuel_pop(x)
        
        out["F"] = [f1, f2, f3]
        
        #Constraints
        
        #Maximum fuel
        g1 = f3 - self.max_fuel
        
        out["G"] = [g1]
        
    def _calc_network_performance_indicator(self, xs):
        performances = []
        
        for x, section in zip(xs, self.section_optimization.values()):
            if x >= len(section['Performance']):
                x = int(x - len(section['Performance']))
            
            performances.append(section['Performance'][x])
        
        return np.mean(performances)
    
    def _calc_network_budget_pop(self, xs):
        sum_costs = []
        
        for x in xs:
            sum_costs.append(self._calc_network_budget(x))
        
        return sum_costs
    
    def _calc_network_budget(self, xs):
        costs = []
        
        for x, section in zip(xs, self.section_optimization):
            actions = self.get_actions(x, section)
            
            for action in actions:
                for item in self.actions:
                    if item['name'] == action:
                        action = self.get_action_option(x, action)
                        multiplyer = self._get_network_multiplyer(self.TMS_output[section], action)
                        cost = multiplyer * item['cost']
                        costs.append(cost)
                        break
        
        return np.sum(costs)
    
    def _get_network_multiplyer(self, section, action):
        return section[action]['Cost']
    
    def _calc_fuel_pop(self, xs):
        fuels = []
        
        for x in xs:
            fuels.append(self._calc_fuel(x))
        
        return np.array(fuels)
    
    def _calc_fuel(self, xs):
        fuel = []
        
        for x, section in zip(xs, self.section_optimization):
            actions = self.get_actions(x, section)
            actions = [self.get_action_option(x, action) for action in actions]
            fuel.append(self.get_fuel(self.TMS_output[section], actions))
            
        return np.sum(fuel)
    
    def get_fuel(self, section, actions):
        fuel = 0
        
        for action in actions:
            fuel += section[action]['Fuel']
        
        return fuel
        
    def get_actions(self, x, section):
        if x >= len(self.section_optimization[section]['Performance']):
            _x = int(x - len(self.section_optimization[section]['Performance']))
            actions = self.section_optimization[section]['Actions_schedule'][_x].values()
        else:
            actions = self.section_optimization[section]['Actions_schedule'][x].values()
        return actions
    
    
    def get_action_option(self, x, action):
        if x > len(self.xu)/2:
            action += '_2'
        else:
            action += '_10'
        return action