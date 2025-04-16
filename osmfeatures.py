
import osmnx as ox
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame


def get_osm_data(coords: tuple[float, float], radius: int, tags: dict):
    """
    Fetch OpenStreetMap data for a given coordinate and radius.

    Args:
        coords (tuple): Latitude and longitude of the center point.
        radius (int): Radius in meters around the center point.
        tags (dict): OSM tags to filter data (e.g., {'amenity': True} for POIs).

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

def get_osm_features(coords: tuple[float, float], radius: int):
    tags = {
        'highway': ['footway', 'cycleway', 'crossing', 'stop', 'traffic_signals'],
        'amenity': True,
        'public_transport': True,
        'landuse': ['retail'],
        'cycleway': True
    }

    feature_data = get_osm_data(coords, radius, tags=tags)
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
        'retail': feature_data[feature_data['landuse'].isin(['retail'])] if 'landuse' in feature_data else None
    }

    densities = {}
    for feature, gdf in features.items():
        densities[feature] = calculate_density(gdf, radius)

    return densities

# coords = (lat, long)
coords4 = (40.7128, -74.0060)  # New York City
coords1 = (33.0786, -96.7047)  # Home Plano
coords2 = (42.3563, -71.0588)
coords3 = (33.360763, -100.110555)

#result = get_osm_features((40.9075, -74.2526), 1000)

#print(result)
