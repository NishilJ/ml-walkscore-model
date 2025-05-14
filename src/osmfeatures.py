from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox


ox.settings.log_console = True


def fetch_osm_as_geojson(coords, radius, tags):
    # Fetch osm data and return as GeoJSON + CRS.
    gdf = ox.features.features_from_point(coords, dist=radius * 1609.344, tags=tags)
    return gdf.to_json(), gdf.crs.to_string() if gdf.crs else None


def get_osm_data(coords, radius, tags, retries=3, timeout=8):
    # Get OSM data for a given coordinate and radius.
    for attempt in range(retries):
        print(f"Attempt {attempt + 1} to fetch OSM data...")

        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_osm_as_geojson, coords, radius, tags)
            try:
                json_data, crs = future.result(timeout=timeout)
                gdf = gpd.read_file(StringIO(json_data))
                if crs:
                    gdf.set_crs(crs, inplace=True)
                return gdf

            except TimeoutError:
                print(f"Timeout on attempt {attempt + 1}. Retrying...")
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")

    print("All attempts failed. Returning empty GeoDataFrame.")
    return gpd.GeoDataFrame()


def calculate_density(gdf: pd.DataFrame, radius: float):
    # Calculate the density of features per square mile.
    if gdf is None or gdf.empty or radius == 0:
        return 0
    return round(len(gdf) / (np.pi * radius ** 2), 2)


def get_osm_feature_densities(coords: tuple[float, float], radius: float):
    tags = {
        'highway': ['footway', 'cycleway', 'crossing', 'stop', 'traffic_signals'],
        'amenity': True,
        'public_transport': True,
        'cycleway': True,
    }
    print(f"Retrieving OSM feature data at {coords}...")
    feature_data = get_osm_data(coords, radius, tags)
    features = {
        'intersections': feature_data[
            feature_data['highway'].isin(['stop', 'traffic_signals'])] if 'highway' in feature_data else None,
        'pedways': feature_data[
            feature_data['highway'].isin(['footway', 'crossing'])] if 'highway' in feature_data else None,
        'bikeways':
            pd.concat([
                feature_data[
                    feature_data['highway'].isin(['cycleway'])] if 'highway' in feature_data else pd.DataFrame(),
                feature_data[feature_data['cycleway'].notna()] if 'cycleway' in feature_data else pd.DataFrame()
            ], ignore_index=True),
        'pois': feature_data[feature_data['amenity'].notna()] if 'amenity' in feature_data else None,
        'transit_stops': feature_data[
            feature_data['public_transport'].notna()] if 'public_transport' in feature_data else None,
    }

    densities = {}
    for feature, gdf in features.items():
        densities[feature] = calculate_density(gdf, radius)
    print(f"OSM feature densities successfully calculated at {coords}...")
    return densities
