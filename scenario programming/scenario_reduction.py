"""
3  # @File  : scenario_reduction.py
4  # @Author: Chen Zhen
5  # @Date  : 2019/9/17
6  # @Desc  :  reduce the number of scenarios based on the paper: A two stage stochastic programming model for 
               lot-sizing and scheduling under uncertainty (2016) in CIE.
               
               demand follow non-stationary Weibull distribution.
               single-item, three periods:
               mean = [467, 34, 150]
               variance = [99422, 175, 4878]
               skew = [1.06, 0.25, 0.47]
               kurt = [4.35 2.78 2.98]
               
             
"""

# in each period there are five scenarios obtained by Lingo

demand1_scenarios = [469.0991593, 509.3416195, 78.44273046, 1248.680927, 521.1852173]
demand1_probs = [0.259515204, 0.291146581, 0.2483824, 0.1, 0.100955815]

demand2_scenarios = [34.12093184, 40.90651414, 40.80139453, 61.31685471, 15.8583253]
demand2_probs = [0.35, 0.145936376, 0.140957808, 0.100715976, 0.26238984]

demand3_scenarios = [116.5985386, 190.6280441, 31.61344284, 103.5003829, 298.9006897]
demand3_probs = [0.278719043, 0.35, 0.1, 0.171266148, 0.100014808]

scenario_num_need = 10  # scenario number after reducing


