import osmnx as ox
import numpy as np


def get_osm_data(coords, radius, tags):
    """
    Fetch OpenStreetMap data for a given coordinate and radius.

    Args:
        coords (tuple): Latitude and longitude of the center point.
        radius (int): Radius in meters around the center point.
        tags (dict): OSM tags to filter data
    Returns:
        GeoDataFrame: OSM data within the specified area.
    """
    try:
        return ox.features.features_from_point(coords, dist=radius, tags=tags)
    except:
        return None


def calculate_density(gdf, area):
    """
    Calculate the density of features per square kilometer.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the features.
        area (float): Area in square kilometers.

    Returns:
        float: Density of features per square kilometer.
    """
    if gdf is None or area == 0:
        return 0
    return len(gdf) / area


def main():
    coords = (40.7128, -74.0060)  # New York City
    coords1 = (33.0786, -96.7047)  # Home Plano
    radius = 500  # meters

    # Calculate the area of the circular region in square kilometers
    area = np.pi * (radius ** 2) / 1e6

    # Get intersection density
    intersections = get_osm_data(coords, radius, tags={'highway': [ 'traffic_signals', 'stops']})
    intersection_density = calculate_density(intersections, area)

    # Get pedestrian infrastructure density
    pedways = get_osm_data(coords, radius, tags={'highway': ['footway', 'crossing'],
                                                 'footway': ['sidewalk', 'crossing', 'traffic_island']})
    general_footways = pedways[
        (pedways['highway'] == 'footway') & (~pedways['footway'].isin(['sidewalk', 'crossing', 'traffic_island']))]
    sidewalks = pedways[pedways['footway'] == 'sidewalk']
    crossings = pedways[pedways['footway'] == 'crossing']
    general_footway_density = calculate_density(general_footways, area)
    sidewalk_density = calculate_density(sidewalks, area)
    crossing_density = calculate_density(crossings, area)
    pedway_density = general_footway_density + sidewalk_density + crossing_density

    # Get bike infrastructure density
    bikeways = get_osm_data(coords, radius, tags={'highway': ['cycleway']})
    bikeway_density = calculate_density(bikeways, area)

    # Get POI density (e.g., amenities like restaurants, shops, etc.)
    pois = get_osm_data(coords, radius, tags={'amenity': True})
    poi_density = calculate_density(pois, area)

    # Get transit stop density (e.g., bus stops, train stations)
    transit_stops = get_osm_data(coords, radius, tags={'public_transport': ['stop_position', 'platform']})
    transit_density = calculate_density(transit_stops, area)

    print(f"Intersection Density: {intersection_density:.2f} per km²")
    print(f"Pedway Density: {pedway_density:.2f} per km²")
    print(f"Bikeway Density: {bikeway_density:.2f} per km²")
    print(f"POI Density: {poi_density:.2f} per km²")
    print(f"Transit Stop Density: {transit_density:.2f} per km²")


if __name__ == "__main__":
    main()
