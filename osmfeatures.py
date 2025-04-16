import osmnx as ox
import numpy as np
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
        return None


def calculate_density(gdf : GeoDataFrame, radius: int) :
    """
    Calculate the density of features per square kilometer.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the features.
        radius (int): Radius in meters around the center point.

    Returns:
        float: Density of features per square kilometer.
    """
    if gdf is None or  radius == 0:
        return 0.00
    return round(len(gdf) * 1e6 / (np.pi * radius**2), 2)

def get_osm_features(coords: tuple[float, float], radius: int):
    features = {
        'intersections': get_osm_data(coords, radius, tags={'highway': ['stop', 'traffic_signals']}),
        'pedways': get_osm_data(coords, radius, tags={'highway': ['footway', 'crossing']}),
        'bikeways': get_osm_data(coords, radius, tags={'highway': ['cycleway'], 'cycleway': True}),
        'pois': get_osm_data(coords, radius, tags={'amenity': True}),
        'transit_stops': get_osm_data(coords, radius, tags={'public_transport': True}),
        'retail': get_osm_data(coords, radius, tags={'landuse': ['retail']})
    }

    densities = {}
    for feature, gdf in features.items():
        densities[feature] = calculate_density(gdf, radius)

    return densities

# coords = (lat, long)
#coords = (40.7128, -74.0060)  # New York City
#coords2 = (33.0786, -96.7047)  # Home Plano
#coords2 = (42.3563, -71.0588)

