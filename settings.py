# -*- coding: utf-8 -*-

# This files highlights all the necessary configurations of  the environment
# during training. To adjust the params, make sure you've checked the 
# instructions. An inappropriate modification may cause error.
# Note: Do not hashtag any exsisting params! 

#  Traffic light controller acitons.
# See https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html
Actions = {
    "W, E": "grrgrrrgGrgGGGGrrgrrgrrrgGrgGGGGrGrGr",
    "WL, EL": "grrgrrrgrGgrrrrGGgrrgrrrgrGgrrrrGrrrr",
    "W, WL": "grrgrrrgrrgrrrrrrgrrgrrrgGGgGGGGGrrrr",
    "E, EL": "grrgrrrgGGgGGGGGGgrrgrrrgrrgrrrrrrrrr",
    "N, S":"gGrgGGrgrrgrrrrrrgGrgGGrgrrgrrrrrrGrG",
    "NL, SL": "grGgrrGgrrgrrrrrrgrGgrrGgrrgrrrrrrrrr",
    "N, NL": "gGGgGGGgrrgrrrrrrgrrgrrrgrrgrrrrrrrrr",
    "S, SL": "grrgrrrgrrgrrrrrrgGGgGGGgrrgrrrrrrrrr"
}

#  Observation area
# The area is describe using four of its boundaries: [L, R, D, U]
Area = [-55, 65, -30, 78]

# Frameskip (in frames)
Frameskip = 75

# Reward range
Reward_range = None

# Warm up durations (in secs)
Warm_up_time = 60

# Simulation durations (in secs)
Total_time = 3600

# Sumo interactive env params.
# See https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
Edges = [
    "CAED","CAEE","CAEU","CAEX",
    "CAWD","CAWE","CAWU","CAWX",
    "JSND","JSNE","JSNU","JSNX",
    "JSSD","JSSE","JSSU","JSSX"
    ]
Entrance = [
    "CAEE", "CAWE", "JSNE", "JSSE"
]
Sumobinary = "sumo-gui"
Sumopath = "./sumo/ITLS.sumo.cfg"
TLid = "ITLS"

# Conversion factors of vehicles
# See http://www.mohurd.gov.cn/wjfb/201903/t20190320_239844.html
Conversions = {
    "NMV": 0.36,
    "P": 1.0,
    "T": 1.2,
    "MP": 2.0,
    "MT": 3.0,
    "LP": 2.5,
    "LT": 4.0
}