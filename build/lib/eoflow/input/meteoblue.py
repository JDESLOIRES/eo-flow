import requests as req
import pandas as pd
import time
from datetime import datetime
import io


class CEHUB_Extraction:
    def __init__(self, queryBackbone):

        self.queryBackbone = queryBackbone

    def format_query(self, start_time, end_time, coordinates=[(46, 0.5)], location_names=["test"]):
        self.queryBackbone["geometry"]["geometries"] = \
            [dict(type='MultiPoint', coordinates=coordinates, locationNames=location_names)]
        self.queryBackbone["timeIntervals"] = [start_time + 'T+00:00' + '/' + end_time + 'T+00:00']

    @staticmethod
    def validate(date_text):
        try:
            datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

    @staticmethod
    def url_request(jobID):
        """
        download the result dataset and return a DataFrame
        if the result dataset is in csv format
        (e.g. it exceed the rows limit in Excel)
        then the raw text will be fixed

        input: jobID from request sent to server, see function «inviare» below
        """
        rawBit = req.get("http://queueresults.meteoblue.com/" + jobID)  # HTTPS not yet available
        urlData = rawBit.content
        return pd.read_csv(io.StringIO(urlData.decode('utf-8')))

    def request_df(self, query, id_request, start_time, end_time, coordinates=None, rsuffix='mean',
                   location_names=["test"]):
        """
        send requests to meteoblue server, receive responses and handle error

        input: query in list format, see API syntax
        """
        if coordinates is None :
            coordinates = [(46, 0.5)] #just for check
        self.validate(start_time)
        self.validate(end_time)

        self.format_query(start_time, end_time, coordinates, location_names)
        self.queryBackbone["queries"] = query

        response = req.post("http://my.meteoblue.com/dataset/query",  # HTTPS not yet available
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                            params={"apikey": "syn23wrpuwencie"},
                            # ask the person in charge for an API key or use this one
                            json=self.queryBackbone
                            )

        if response.status_code != 200:
            raise Exception(response.json()["error_message"])
        jobID = response.json()["id"]
        print(jobID)
        jobStatus = req.get("http://my.meteoblue.com/queue/status/" + jobID).json()[
            "status"]  # HTTPS not yet available
        print(jobStatus)
        while jobStatus == "running":
            time.sleep(20)  # pause
            jobStatus = req.get("http://my.meteoblue.com/queue/status/" + jobID).json()["status"]
            print(jobStatus)
        if jobStatus != "finished":
            raise Exception("unexpected error: relaunch query to see whether the problem persists")
        print("converting to DataFrame")
        return self.url_request(jobID)


'''
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

query = [{"domain": "ERA5", "gapFillDomain": "NEMSGLOBAL", "timeResolution": "daily",
            "codes": [ # *derived → with caution
                    {"code":  11, "level": "2 m above gnd","aggregation": "mean"}, # air temperature (°C)
                    {"code":  17, "level": "2 m above gnd","aggregation": "mean"}, # Dewpoint Temperature
                    {"code":  32, "level": "2 m above gnd","aggregation": "mean"}, # Wind Speed
                    {"code":  52, "level": "2 m above gnd","aggregation": "mean"}, # Relative Humidity
                    #{"code":  56, "level": "2 m above gnd"}, # Vapor Pressure Deficit
                    #{"code":  61, "level": "sfc"}, # Precipitation Total
                    {"code": 71, "level": "sfc","aggregation": "mean"}, # Cloud Cover Total
                    #{"code": 191, "level": "sfc"}, # Sunshine Duration
                    #{"code":  256, "level": "sfc"}, # Diffuse Shortwave Radiation
                    #{"code":  260, "level": "2 m above gnd"}, # FAO Reference Evapotranspiration
                    {"code": 1, "level": "2 m above gnd","aggregation": "mean"}, # Pressure
                    {"code": 730, "level": "2 m above gnd","aggregation": "sum","gddBase": 8,
                     "gddLimit": 30}, # Growing Degree Days
                    ],
}]

#Query soil
{
    "units": {
        "temperature": "C",
        "velocity": "km/h",
        "length": "metric",
        "energy": "watts"
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [
                    1.959448,
                    43.685787
                ],
                [
                    1.950384,
                    43.678636
                ],
                [
                    1.964392,
                    43.671584
                ],
                [
                    1.971945,
                    43.674365
                ],
                [
                    1.959448,
                    43.685787
                ]
            ]
        ]
    },
    "format": "json",
    "timeIntervals": [
        "2020-01-01T+00:00/2020-12-31T+00:00"
    ],
    "timeIntervalsAlignment": "none",
    "queries": [
        {
            "domain": "SOILGRIDS2",
            "gapFillDomain": null,
            "timeResolution": "static",
            "codes": [
                { "code": 808, "level": "aggregated", "startDepth": 0, "endDepth": 150}, #bulk
                {"code": 809, "level": "aggregated", "startDepth": 0, "endDepth": 150 },


                ]

                }
            ]
        }
    ]
}



# import the libraries
import requests
import json

# define the api-endpoint
myAPIendpoint = "https://cropfact.syngentaaws.org/services/cropcalendar"

# define the header
myHeader = {'x-api-key' : '5Bm6tpWR1Taf27uUT0EM61LlODvCjrOm30vqCrKN',
            'cehub-api-key' : 'syng63gdwiuhiudw',
            'Content-Type' : 'application/json'}

# define the body
myJson_txt = (
    '{'
    '   "scenario": {'
    '       "croppingArea": {'
    '           "country": "US",'
    '           "geometry": {'
    '               "type": "Point",'
    '               "coordinates": [-89.28204, 40.38502]'
    '           }'
    '       },'
    '       "genotype": {'
    '           "crop": "Corn Grain"'
    '       },'
    '       "management": {'
    '            "season": "1",'
    '            "plantingTime": "2020-02-01",'
    '            "harvestTime": "2020-09-10"'
    '       }'
    '   }'
    '}'
)
myBody = json.loads(myJson_txt)

# send post-request and save response as response object
myResponse = requests.post(url=myAPIendpoint, headers=myHeader, json=myBody)
myResponse.json()
'''
