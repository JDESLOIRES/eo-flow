import pandas as pd
from datetime import datetime
import aiohttp
import asyncio
from io import StringIO

#https://stackoverflow.com/questions/42009202/how-to-call-a-async-function-contained-in-a-class

'''
Les nœuds de demande asynchrones rendent le contrôle au flux sans attendre de réponse. Cette action libère le thread de requête pour gérer d'autres requêtes, 
tandis que la réponse est gérée par le nœud de réponse apparié sur un thread différent et dans une nouvelle transaction.
'''
#https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp


def format_query(queryBackbone,  query,
                 start_time, end_time,
                 coordinates, location_names):
    '''
    :param queryBackbone (dict)
    :param query (dict)
    :param start_time (str) : yyyy-mm-dd
    :param end_time (str) : yyyy-mm-dd
    :param coordinates : list
    :param location_names: list
    :return: dictionary with queryBackbone updated w.r.t start_time, end_time, coordinates and locations
    '''
    queryBackbone["geometry"]["geometries"] = \
        [dict(type='MultiPoint', coordinates=coordinates, locationNames=location_names)]
    queryBackbone["timeIntervals"] = [start_time + 'T+00:00' + '/' + end_time + 'T+00:00']
    queryBackbone["queries"] = query
    return queryBackbone


def validate_time_interval(date_text):
    '''
    Check if date input is in the right format
    :param date_text (str)
    :return:
    '''
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


async def get_jobID(session, queryBackbone, key, sleep = 1):
    '''

    :param session:
    :param queryBackbone:
    :param key:
    :param sleep:
    :return:
    '''
    await asyncio.sleep(sleep)
    async with session.post("http://my.meteoblue.com/dataset/query",
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                            params={"apikey": key},
                            json=queryBackbone
                            ) as response:
        data = await response.json()
        print(data)
        return data['id']



async def get_jobIDs_from_query(queryBackbone, query, ids, coordinates, years, key, time_interval = ('03-30', '11-25'), sleep = 3):
    '''
    :param queryBackbone:
    :param query:
    :param ids:
    :param coordinates:
    :param years:
    :param key:
    :param time_interval:
    :param sleep:
    :return:
    '''

    async def make_ids(ids, coordinates, dates):
        for i, (id, coord, date) in enumerate(zip(ids, coordinates, dates)):
            yield i, id, coord, date

    async with aiohttp.ClientSession() as session:
        post_tasks = []
        # prepare the coroutines that post
        async for i, id, coord, date in make_ids(ids, coordinates, years):
            start_time, end_time = (str(date) + "-" + time_interval[0], str(date) + "-" + time_interval[0])
            print(start_time)
            queryBackbone = format_query(queryBackbone, query,
                                         start_time, end_time,
                                         coordinates=[coord],
                                         location_names=[id])
            post_tasks.append(get_jobID(session, queryBackbone, key, (sleep * i)))
        # now execute them all at once
        jobIDs = await asyncio.gather(*post_tasks)
        return jobIDs



async def get_request_from_jobID(jobID, sleep = 1, limit = None):
    '''
    :param jobID:
    :param sleep:
    :param limit:
    :return:
    '''
    await asyncio.sleep(sleep)
    conn = aiohttp.TCPConnector(limit=limit, ttl_dns_cache=300)
    session = aiohttp.ClientSession(connector=conn)

    async with session.get("http://queueresults.meteoblue.com/" + jobID) as response:
        print("Status:", response.status)
        print("Content-type:", response.headers['content-type'])
        urlData = await response.text()
        print(response)
        await session.close()
    return pd.read_csv(StringIO(urlData), sep=",", header=None)


async def get_dataframe_from_jobIDs(jobsIDs):
    '''
    Define dataframe with time series data from the job ids obtained from getting http request
    :param jobsIDs:
    :return:
    '''
    async def make_jobs(jobIDs):
        for i, x in enumerate(jobIDs):
            yield i, x
    async with aiohttp.ClientSession() as session:
        post_tasks = []
        # prepare the coroutines that post
        async for i, jobID in make_jobs(jobsIDs):
            post_tasks.append(get_request_from_jobID(jobID, i))
        # now execute them all at once
        dfs = await asyncio.gather(*post_tasks)
        return dfs


async def gather_with_concurrency(n, *tasks):
    semaphore = asyncio.Semaphore(n)
    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))



#############################################################################################################
#############################################################################################################
