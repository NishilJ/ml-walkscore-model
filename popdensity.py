from urllib.parse import urlencode

import aiohttp
import asyncio


async def fetch_with_retries(url, params, retries=10, delay=3):
    for attempt in range(retries):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 502:
                        print(f"502 Error: Retrying {attempt + 1}/{retries} in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue  # Retry
                    response.raise_for_status()  # Raise error for other issues
                    return await response.json()
            except aiohttp.ClientError as e:
                print(f"Request failed: {e}")
                return None
    print("Max retries reached. Returning None.")
    return None

async def get_geoid(coords: tuple[float, float]):
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        "x": coords[1],
        "y": coords[0],
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "layers": "8",
        "format": "json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params):
            try:
                data = await fetch_with_retries(url, params)
                gid = data["result"]["geographies"]["Census Tracts"][0]["GEOID"]
                print(f"GEOID: {gid} for {coords} successfully retrieved from {url}?{urlencode(params)}")
                return gid
            except TypeError:
                print(f"ERROR getting GEOID for Coords: {coords}")
                return None


async def get_tract_population(gid):
    """Fetches total population for a given Census GEOID using ACS 5-Year Data."""
    url = "https://api.census.gov/data/2023/acs/acs5"
    params = {
        "get": "B01003_001E",  # Total population variable
        "ucgid": f"1400000US{gid}"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            try:
                data = await response.json()
                pop =  int(data[1][0])
                print(f"Pop: {pop} for GID: {gid} successfully retrieved from {url}?{urlencode(params)}")
                return pop
            except (KeyError, IndexError, ValueError, TypeError):
                print(f"ERROR getting population for GEOID: {gid} from {url}?{urlencode(params)}")
                return None


async def get_tract_land(gid):
    """Fetches the land area in square miles for a given Census tract GEOID."""
    url = "https://api.census.gov/data/2023/geoinfo"
    params = {
        "get": "AREALAND_SQMI",
        "ucgid": f"1400000US{gid}"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            try:
                data = await response.json()
                land = float(data[1][0])
                print(f"Land: {land} for GID: {gid} successfully retrieved from {url}?{urlencode(params)}")
                return land
            except TypeError:
                print(f"ERROR getting land area for GEOID: {gid} from {url}?{urlencode(params)}")
                return None


async def get_pop_density(coords: tuple[float, float]):
    gid = await get_geoid(coords)
    pop = await get_tract_population(gid)
    land = await get_tract_land(gid)

    if pop is None or land is None or land == 0:
        print(f"ERROR getting pop density at ({coords})")
        return None
    return round(pop / land, 2)
