import asyncio
import csv
import random
import geopandas as gpd
from shapely.geometry import Point

from popdensity import get_pop_density
from osmfeatures import get_osm_features

data_points = 50 # Total amount of coord data points to generate
radius = 500 # Find OSM features in a radius (meters) around each coord

def get_random_us_coord():
    #us_boundary = gpd.read_file("boundaries/us/ne_110m_admin_0_countries.shp")
    #us_boundary = us_boundary[us_boundary['ADMIN'] == 'United States of America']
    nyc_boundary = gpd.read_file("boundaries/nyc/columbia_nycp_2000_blocks.shp")
    while True:
        lat = random.uniform(40.4, 41.2) # NYC bounding coords
        lon = random.uniform(-74.4, -73.5)
        #lat = random.uniform(32.5, 33.3) # DFW bounding coords
        #lon = random.uniform(-97.9, -96.4)
        #lat = random.uniform(24.5212, 49.3828) # U.S. bounding coords
        #lon = random.uniform(-124.7363, -66.9454)

        point = Point(lon, lat)
        if nyc_boundary.geometry.contains(point).any():
            return round(lat, 4), round(lon, 4)

async def main():
    coords_list = [get_random_us_coord() for _ in range(data_points)] # TOTAL AMOUNT OF COORD ENTRIES

    print("Calculating population densities...")
    pop_densities = await asyncio.gather(*[get_pop_density(coords) for coords in coords_list])
    print("Population densities calculated!")

    file_path = "data.csv"
    header = ['Lat', 'Lon', 'Pop Density', 'Intersections', 'Pedways', 'Bikeways', 'POIs', 'Transit', 'Retail']
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)

    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f, delimiter=',')

        for i, coords in enumerate(coords_list):
            pop_density = pop_densities[i]
            print(f"#{i + 1} Calculating OSM feature densities at ({coords})...")
            osm_features = get_osm_features(coords, radius)

            row = [coords[0], coords[1], pop_density,
                         osm_features['intersections'],
                         osm_features['pedways'],
                         osm_features['bikeways'],
                         osm_features['pois'],
                         osm_features['transit_stops'],
                         osm_features['retail']]

            writer.writerow(row)
            f.flush()
            print(f"#{i + 1} {coords} row written to {file_path}.")

    print(f"Data generation complete at {file_path}.")

if __name__ == "__main__":
    asyncio.run(main())
