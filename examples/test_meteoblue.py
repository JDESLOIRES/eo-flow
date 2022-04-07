import pandas as pd
import asyncio
from ast import literal_eval
import eoflow.input.meteoblue as meteoblue
import pickle
import os


path_files = '/home/johann/Documents/Syngenta/shapefile'
input_file = pd.read_csv(os.path.join(path_files, 'input_shapefile.csv'))
input_file['coordinates'] = [literal_eval(k) for k in input_file['coordinates'].values]

###########################################################################################

queryBackbone = {
    "units": {"temperature": "C", "velocity": "m/s", "length": "metric"},
    "timeIntervalsAlignment": None,
    "runOnJobQueue": True,
    "oneTimeIntervalPerGeometry": True,
    "checkOnly": False,
    "requiresJobQueue": False,
    "geometry": {
        "type": "GeometryCollection",
        "geometries": None
    },
    "format": "csvIrregular", # best format
    "timeIntervals":  None
}

#Query weather
stat = 'mean'

query = [{"domain": "ERA5", "gapFillDomain": "NEMSGLOBAL",
          "timeResolution": "daily",
          "codes": [
              {"code":  11, "level": "2 m above gnd", "aggregation": stat}, # air temperature (°C)
              {"code":  32, "level": "2 m above gnd", "aggregation": stat}, # Wind Speed
              #{"code":180, "aggregation": stat}, #Wind Gust
              {"code":  256, "level": "sfc","aggregation": stat}, # Diffuse Shortwave Radiation
              {"code":  56, "level": "2 m above gnd","aggregation": stat}, # Vapor Pressure Deficit
              {"code":  260, "level": "2 m above gnd","aggregation": stat}, # FAO Reference Evapotranspiration,
              {"code":  261, "level": "sfc", "aggregation": stat}, # Evapotranspiration
          ],
}]

query_sum = [{"domain": "ERA5", "gapFillDomain": "NEMSGLOBAL",
          "timeResolution": "daily",
          "codes": [
              {"code":  61, "level": "sfc", "aggregation": "sum"}, # Precipitation Total
              {"code":  730, "level": "2 m above gnd", "aggregation": "sum", "gddBase": 8, "gddLimit": 30},
          ],
}]

conc_req = 10
loop= asyncio.new_event_loop()

try:
    jobIDs = loop.run_until_complete(meteoblue.get_jobIDs_from_query(queryBackbone=queryBackbone,
                                                                     query = query,
                                                                     ids = input_file.key.values,
                                                                     coordinates=input_file.coordinates.values,
                                                                     years = input_file.year.values,
                                                                     key = ''))
    dfs = loop.run_until_complete(meteoblue.gather_with_concurrency(conc_req,
                                                                    *[meteoblue.get_request_from_jobID(jobID, i / 100)
                                                                      for i, jobID in enumerate(jobIDs)]))

finally:
    print('close')
    loop.close()

output = pd.concat(dfs, axis = 0)
output.to_csv(os.path.join(path_files, stat + '_weather.csv'), index = False)

