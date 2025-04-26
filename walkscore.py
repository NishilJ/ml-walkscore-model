from urllib.parse import urlencode
import aiohttp

API_KEY = "fcb3be41c472b613725aefc8ee8afebf"

async def get_walk_score(coords: tuple[float, float]):
    """Fetches the WalkScore for a given coordinates."""
    url = "https://api.walkscore.com/score"
    params = {
        "format": "json",
        "lat": coords[0],
        "lon": coords[1],
        "wsapikey": API_KEY,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            try:
                data = await response.json()
                wscore = data["walkscore"]
                print(f"WalkScore: {wscore} for {coords} successfully retrieved from {url}?{urlencode(params)}")
                return wscore
            except:
                print(f"No WalkScore for : {coords} from {url}?{urlencode(params)}")
                return None