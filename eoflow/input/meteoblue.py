import copy

import pandas as pd
from datetime import datetime
import aiohttp
import asyncio
from io import StringIO

# https://stackoverflow.com/questions/42009202/how-to-call-a-async-function-contained-in-a-class
import json

"""
Les nœuds de demande asynchrones rendent le contrôle au flux sans attendre de réponse. Cette action libère le thread de requête pour gérer d'autres requêtes, 
tandis que la réponse est gérée par le nœud de réponse apparié sur un thread différent et dans une nouvelle transaction.
"""
# https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp
# https://stackoverflow.com/questions/42009202/how-to-call-a-async-function-contained-in-a-class


async def get_jobID(queryBackbone, key, sleep=0.5):
    """
    :param session:
    :param queryBackbone:
    :param key:
    :param sleep:
    :return:
    """
    await asyncio.sleep(sleep)
    print(queryBackbone)
    async with aiohttp.ClientSession() as session:
        # prepare the coroutines that post
        async with session.post(
            "http://my.meteoblue.com/dataset/query",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            params={"apikey": key},
            json=queryBackbone,
        ) as response:
            data = await response.json()
            print(data)
        await session.close()
    return data["id"]


async def get_jobIDs_from_query(
    queryBackbone, query, ids, coordinates, years, key, time_interval=("03-30", "11-25")
):
    """
    :param queryBackbone:
    :param query:
    :param ids:
    :param coordinates:
    :param years:
    :param key:
    :param time_interval:
    :param sleep:
    :return:
    """

    async def make_ids(ids, coordinates, dates):
        for i, (id, coord, date) in enumerate(zip(ids, coordinates, dates)):
            yield i, id, coord, date

    jobIDs = []

    async for i, id, coord, date in make_ids(ids, coordinates, years):
        await asyncio.sleep(0.2)
        start_time, end_time = (
            str(date) + "-" + time_interval[0],
            str(date) + "-" + time_interval[1],
        )

        queryBackbone["geometry"]["geometries"] = [
            dict(type="MultiPoint", coordinates=[coord], locationNames=[id])
        ]
        queryBackbone["timeIntervals"] = [
            start_time + "T+00:00" + "/" + end_time + "T+00:00"
        ]
        queryBackbone["queries"] = query

        async with aiohttp.ClientSession() as session:
            # prepare the coroutines that post
            async with session.post(
                "http://my.meteoblue.com/dataset/query",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                params={"apikey": key},
                json=queryBackbone,
            ) as response:
                data = await response.json()
                print(data)
            await session.close()
        jobIDs.append(data["id"])
    # now execute them all at once
    return jobIDs


async def get_request_from_jobID(jobID, sleep=1, limit=None):
    """
    :param jobID:
    :param sleep:
    :param limit:
    :return:
    """
    await asyncio.sleep(sleep)
    # limit amount of simultaneously opened connections you can pass limit parameter to connector
    conn = aiohttp.TCPConnector(limit=limit, ttl_dns_cache=300)
    session = aiohttp.ClientSession(
        connector=conn
    )  # ClientSession is the heart and the main entry point for all client API operations.
    # session contains a cookie storage and connection pool, thus cookies and connections are shared between HTTP requests sent by the same session.

    async with session.get("http://queueresults.meteoblue.com/" + jobID) as response:
        print("Status:", response.status)
        print("Content-type:", response.headers["content-type"])
        urlData = await response.text()
        print(response)
        await session.close()
    df = pd.read_csv(StringIO(urlData), sep=",", header=None)
    df["jobID"] = jobID
    return df


async def gather_with_concurrency(n, *tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))
