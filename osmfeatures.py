import osmnx as ox
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
import asyncio
import time
import aiohttp
import requests


def get_osm_data(coords: tuple[float, float], radius: int, tags: dict):
    """
    Fetch OpenStreetMap data for a given coordinate and radius.

    Args:
        coords (tuple): Latitude and longitude of the center point.
        radius (int): Radius in meters around the center point.
        tags (dict): OSM tags to filter data (ex. {'amenity': True}).

    Returns:
        GeoDataFrame: OSM data within the specified area.
    """
    try:
        return ox.features.features_from_point(coords, dist=radius, tags=tags)
    except:
        return {}


def calculate_density(gdf : pd.DataFrame, radius: int) :
    """
    Calculate the density of features per square kilometer.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the features.
        radius (int): Radius in meters around the center point.

    Returns:
        float: Density of features per square kilometer.
    """
    if gdf is None or radius == 0:
        return 0.00
    return round(len(gdf) * 1e6 / (np.pi * radius**2), 2)

def get_osm_feature_densities(coords: tuple[float, float], radius: int):
    tags = {
        'highway': ['footway', 'cycleway', 'crossing', 'stop', 'traffic_signals'],
        'amenity': True,
        'public_transport': True,
        'cycleway': True,
    }
    print(f"Retrieving OSM feature data at {coords}...")
    feature_data = get_osm_data(coords, radius, tags)
    features = {
        'intersections': feature_data[feature_data['highway'].isin(['stop', 'traffic_signals'])] if 'highway' in feature_data else None,
        'pedways': feature_data[feature_data['highway'].isin(['footway', 'crossing'])] if 'highway' in feature_data else None,
        'bikeways':
                pd.concat([
                    feature_data[feature_data['highway'].isin(['cycleway'])] if 'highway' in feature_data else pd.DataFrame(),
                    feature_data[feature_data['cycleway'].notna()] if 'cycleway' in feature_data else pd.DataFrame()
                ], ignore_index=True),
        'pois': feature_data[feature_data['amenity'].notna()] if 'amenity' in feature_data else None,
        'transit_stops': feature_data[feature_data['public_transport'].notna()] if 'public_transport' in feature_data else None,
    }

    densities = {}
    for feature, gdf in features.items():
        densities[feature] = calculate_density(gdf, radius)
    print(f"OSM feature densities successfully calculated at {coords}...")
    return densities


# coords = (lat, long)
#coords4 = (40.7128, -74.0060)  # New York City
#coords1 = (33.0786, -96.7047)  # Home Plano
#coords2 = (42.3563, -71.0588)
#coords3 = (33.360763, -100.110555)
#coords_list = [coords1]
#print(asyncio.run(get_osm_feature_densities((40.7128, -74.0060), 1000)))
#response = requests.get("https://overpass-api.de/api/status")
#print(response.text)

