import asyncio
import csv
import random
import geopandas as gpd
from shapely.geometry import Point

import logging

logging.basicConfig(level=logging.DEBUG)

from popdensity import get_pop_density
from walkscore import get_walk_score
from osmfeatures import get_osm_feature_densities

total_examples = 10  # Total amount of coord data points to generate
radius = 1000  # Find OSM features in a radius (meters) around each coord
file_path = "data.csv"

nyc_boundary = gpd.read_file("boundaries/nyc/columbia_nycp_2000_blocks.shp")
#us_boundary = gpd.read_file("boundaries/us/ne_110m_admin_0_countries.shp")
#us_boundary = us_boundary[us_boundary['ADMIN'] == 'United States of America']

async def get_random_us_coord():
    while True:
        lat = random.uniform(40.4, 41.2)  # NYC bounding coords
        lon = random.uniform(-74.4, -73.5)
        #lat = random.uniform(32.5, 33.3) # DFW bounding coords
        #lon = random.uniform(-97.9, -96.4)
        #lat = random.uniform(24.5212, 49.3828) # U.S. bounding coords
        #lon = random.uniform(-124.7363, -66.9454)

        point = Point(lon, lat)
        if nyc_boundary.geometry.contains(point).any():
            return round(lat, 4), round(lon, 4)


async def main():
    header = ['Lat', 'Lon', 'Pop Density', 'Intersections', 'Pedways', 'Bikeways', 'POIs', 'Transit', 'WalkScore']

    print("Generating coordinates...")
    coords_list = await asyncio.gather(*[get_random_us_coord() for _ in range(total_examples)])
    print("Generating coordinates: SUCCESS")

    print("Calculating population densities...")
    pop_densities = await asyncio.gather(*[get_pop_density(coords) for coords in coords_list])

    # Get new coord if 0 pop density
    for i in range(len(coords_list)):
        while pop_densities[i] == 0:
            print(f"Population density for {coords_list[i]} is zero, generating a new coordinate...")
            coords_list[i] = await get_random_us_coord()
            pop_densities[i] = await get_pop_density(coords_list[i])
    print("Calculating population densities: SUCCESS")

    print("Calculating WalkScores...")
    walkscore_list = await asyncio.gather(*[get_walk_score(coords) for coords in coords_list])
    print("Calculating WalkScores: SUCCESS")

    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if f.tell() == 0:
            writer.writerow(header)
        for i, coords in enumerate(coords_list):
            pop_density = pop_densities[i]
            osm_features = get_osm_feature_densities(coords, radius)
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
            print(f"#{i + 1} {coords} data written to {file_path}.")
    print("Calculating OSM feature densities: SUCCESS")
    print(f"Data generation complete at {file_path}.")


if __name__ == "__main__":
    asyncio.run(main())
