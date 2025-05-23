import asyncio
import csv
import random
import geopandas as gpd
from shapely.geometry import Point

from popdensity import get_pop_density
from walkscore import get_walk_score
from osmfeatures import get_osm_feature_densities


# IMPORTANT PARAMETERS
BOUNDARY_NAME = "indy"  # Boundary to generate data in
TOTAL_EXAMPLES = 200  # Total amount of data points to generate
RADIUS = 0.5  # Find OSM features in a radius (miles) around each coord
RETURN_FILE_PATH = f"../data/{BOUNDARY_NAME}.csv"  # Which file to add the data entries to, creates a new file if it doesn't exist
BOUNDARIES = {
    "nyc": gpd.read_file("../boundaries/new_york_city/new_york_city.shp"),
    "chicago": gpd.read_file("../boundaries/chicago/chicago.shp"),
    "la": gpd.read_file("../boundaries/los_angeles/los_angeles.shp"),
    "atlanta": gpd.read_file("../boundaries/atlanta/atlanta.shp"),
    "indy": gpd.read_file("../boundaries/indianapolis/indianapolis.shp"),
    "sa": gpd.read_file("../boundaries/san_antonio/san_antonio.shp"),
    "houston": gpd.read_file("../boundaries/houston/houston.shp"),
    "austin": gpd.read_file("../boundaries/austin/austin.shp"),
    "dallas": gpd.read_file("../boundaries/dallas/dallas.shp")
}


async def get_random_us_coord():
    if BOUNDARY_NAME not in BOUNDARIES:
        raise ValueError(f"Unsupported region: {BOUNDARY_NAME}")
    boundary = BOUNDARIES[BOUNDARY_NAME].to_crs(epsg=4326)
    polygon = boundary.union_all()
    minx, miny, maxx, maxy = polygon.bounds

    while True:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):
            return round(random_point.y, 4), round(random_point.x, 4)


async def main():
    print("Generating coordinates...")
    coords_list = await asyncio.gather(*[get_random_us_coord() for _ in range(TOTAL_EXAMPLES)])
    print("Generating coordinates: SUCCESS")

    print("Calculating population densities...")
    pop_densities = await asyncio.gather(*[get_pop_density(coords) for coords in coords_list])
    print("Calculating population densities: SUCCESS")

    print("Calculating WalkScores...")
    walkscore_list = await asyncio.gather(*[get_walk_score(coords) for coords in coords_list])
    print("Calculating WalkScores: SUCCESS")

    # Get new coord and data if pop density = 0
    print("Validating features...")
    ws_retry_counter = 0
    for i in range(len(coords_list)):
        while pop_densities[i] == 0 or pop_densities[i] is None or walkscore_list[i] is None:
            print(f"WS RETRIES: {ws_retry_counter}, Population density is zero or WalkScore is null for {coords_list[i]}, generating a new coordinate and data...")
            coords_list[i] = await get_random_us_coord()
            pop_densities[i] = await get_pop_density(coords_list[i])

            # Only proceed to get a walkscore if population density is valid
            if pop_densities[i] != 0 and pop_densities[i] is not None:
                walkscore_list[i] = await get_walk_score(coords_list[i])
                ws_retry_counter += 1
    print(f"Validating features: SUCCESS, WS RETRIES: {ws_retry_counter}")

    with open(RETURN_FILE_PATH, mode='a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if f.tell() == 0:
            header = ['Lat', 'Lon', 'Pop Density', 'Intersections', 'Pedways', 'Bikeways', 'POIs', 'Transit', 'WalkScore']
            writer.writerow(header)
        for i, coords in enumerate(coords_list):
            pop_density = pop_densities[i]
            osm_features = get_osm_feature_densities(coords, RADIUS)
            walkscore = walkscore_list[i]
            row = [coords[0], coords[1], pop_density,
                   osm_features['intersections'],
                   osm_features['pedways'],
                   osm_features['bikeways'],
                   osm_features['pois'],
                   osm_features['transit_stops'],
                   walkscore]
            writer.writerow(row)
            f.flush()
            print(f"#{i + 1} {coords} data written to {RETURN_FILE_PATH}.")
    print("Calculating OSM feature densities: SUCCESS")
    print(f"Data generation complete at {RETURN_FILE_PATH}.")


if __name__ == "__main__":
    asyncio.run(main())
