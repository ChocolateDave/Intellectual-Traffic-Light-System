# !/usr/bin/env python
# -*- coding: utf-8 -*-
from xml.dom.minidom import parse

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
# Edges = [
#     "CAED","CAEE","CAEU","CAEX",
#     "CAWD","CAWE","CAWU","CAWX",
#     "JSND","JSNE","JSNU","JSNX",
#     "JSSD","JSSE","JSSU","JSSX",
#     "ITLS",
#    ]
Edges = [
    ':ITLS_0', ':ITLS_1', ':ITLS_2', ':ITLS_3', ':ITLS_4', ':ITLS_6',
    ':ITLS_33', ':ITLS_34', ':ITLS_35', ':ITLS_36', ':ITLS_7', ':ITLS_8',
    ':ITLS_9', ':ITLS_10',':ITLS_11', ':ITLS_15', ':ITLS_16', ':ITLS_37',
    ':ITLS_38', ':ITLS_39', ':ITLS_40', ':ITLS_41', ':ITLS_17', ':ITLS_18',
    ':ITLS_19', ':ITLS_20', ':ITLS_21', ':ITLS_23', ':ITLS_42', ':ITLS_43',
    ':ITLS_44', ':ITLS_45', ':ITLS_24', ':ITLS_25', ':ITLS_26', ':ITLS_27',
    ':ITLS_28', ':ITLS_32', ':ITLS_46', ':ITLS_47', ':ITLS_48', ':ITLS_49',
    ':ITLS_c0', ':ITLS_c1', ':ITLS_c2', ':ITLS_c3', ':ITLS_w0', ':ITLS_w1',
    ':ITLS_w2', ':ITLS_w3', ':gneJ19_w0', ':gneJ20_0', ':gneJ20_7', ':gneJ20_w0',
    ':gneJ20_w1', ':gneJ22_0', ':gneJ22_5', ':gneJ22_w0', ':gneJ22_w1', ':gneJ23_w0',
    ':gneJ24_0', ':gneJ24_5', ':gneJ24_w0', ':gneJ24_w1', ':gneJ25_w0', ':gneJ26_0',
    ':gneJ26_3', ':gneJ26_w0', ':gneJ26_w1', ':gneJ27_w0',
    'CAED', 'CAEE', 'CAEU', 'CAEX', 'CAWD', 'CAWE', 'CAWU', 'CAWX', 
    'JSND', 'JSNE', 'JSNU', 'JSNX', 'JSSD', 'JSSE', 'JSSU', 'JSSX'
]
Entrances = [
    'CAEE','CAWE','JSNE','JSSE'
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