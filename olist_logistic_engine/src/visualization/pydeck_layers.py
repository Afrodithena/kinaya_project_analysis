"""
PyDeck layer configurations for Olist Logistics Network Visualization
"""

import pydeck as pdk
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def get_arc_layer(data: pd.DataFrame, 
                  get_width: str = 'line_width',
                  get_color: str = 'color',
                  dash_array: Optional[List[int]] = None,
                  dash_offset: int = 0) -> pdk.Layer:
    """
    Create an arc layer for route visualization between seller and customer locations.
    
    Args:
        data: DataFrame with seller_lat, seller_lng, customer_lat, customer_lng columns
        get_width: Column name for line width
        get_color: Column name for color (RGBA list)
        dash_array: Dash pattern for animated lines [dash_length, gap_length]
        dash_offset: Offset for dash animation
    
    Returns:
        PyDeck ArcLayer
    """
    layer_config = {
        'type': 'ArcLayer',
        'data': data,
        'get_source_position': ['seller_lng', 'seller_lat'],
        'get_target_position': ['customer_lng', 'customer_lat'],
        'get_width': get_width,
        'get_source_color': get_color,
        'get_target_color': get_color,
        'pickable': True,
        'auto_highlight': True,
        'highlight_color': [255, 255, 255, 100],
        'width_scale': 1,
        'width_min_pixels': 1,
        'width_max_pixels': 10,
    }
    
    if dash_array is not None:
        layer_config['get_dash_array'] = dash_array
        layer_config['get_dash_offset'] = dash_offset
    
    return pdk.Layer(**layer_config)


def get_scatter_layer(data: pd.DataFrame,
                      get_radius: str = 'radius',
                      get_color: Optional[List[int]] = None,
                      get_line_color: Optional[List[int]] = None,
                      line_width: int = 1,
                      opacity: float = 0.8) -> pdk.Layer:
    """
    Create a scatterplot layer for state centroids or points.
    
    Args:
        data: DataFrame with lat, lng columns
        get_radius: Column name or value for point radius
        get_color: RGBA color list or column name
        get_line_color: RGBA color for border line
        line_width: Border width in pixels
        opacity: Layer opacity
    
    Returns:
        PyDeck ScatterplotLayer
    """
    if get_color is None:
        get_color = [50, 50, 80, int(255 * opacity)]
    
    layer_config = {
        'type': 'ScatterplotLayer',
        'data': data,
        'get_position': ['lng', 'lat'],
        'get_radius': get_radius,
        'get_fill_color': get_color,
        'pickable': True,
        'opacity': opacity,
        'radius_scale': 1,
        'radius_min_pixels': 3,
        'radius_max_pixels': 100,
    }
    
    if get_line_color is not None:
        layer_config['get_line_color'] = get_line_color
        layer_config['line_width_min_pixels'] = line_width
    
    return pdk.Layer(**layer_config)


def get_warehouse_layer(data: pd.DataFrame,
                        radius: int = 12000,
                        fill_color: Optional[List[int]] = None,
                        line_color: Optional[List[int]] = None) -> pdk.Layer:
    """
    Create a scatterplot layer for warehouse candidate locations.
    
    Args:
        data: DataFrame with lat, lng columns
        radius: Point radius in meters
        fill_color: RGBA fill color
        line_color: RGBA border color
    
    Returns:
        PyDeck ScatterplotLayer for warehouses
    """
    if fill_color is None:
        fill_color = [255, 100, 0, 200]
    
    if line_color is None:
        line_color = [255, 200, 0, 255]
    
    return pdk.Layer(
        'ScatterplotLayer',
        data=data,
        get_position=['lng', 'lat'],
        get_radius=radius,
        get_fill_color=fill_color,
        get_line_color=line_color,
        line_width_min_pixels=3,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 150]
    )


def get_text_layer(data: pd.DataFrame,
                   get_text: str = 'label',
                   get_size: int = 14,
                   get_color: Optional[List[int]] = None,
                   get_anchor: str = 'center') -> pdk.Layer:
    """
    Create a text layer for labels on map.
    
    Args:
        data: DataFrame with lat, lng, and text columns
        get_text: Column name for text content
        get_size: Text size in pixels
        get_color: RGBA color for text
        get_anchor: Text anchor ('start', 'middle', 'end')
    
    Returns:
        PyDeck TextLayer
    """
    if get_color is None:
        get_color = [255, 255, 255, 255]
    
    anchor_map = {
        'start': '"start"',
        'middle': '"middle"',
        'center': '"center"',
        'end': '"end"'
    }
    
    return pdk.Layer(
        'TextLayer',
        data=data,
        get_position=['lng', 'lat'],
        get_text=get_text,
        get_size=get_size,
        get_color=get_color,
        get_angle=0,
        get_text_anchor=anchor_map.get(get_anchor, '"center"'),
        get_alignment_baseline='"center"',
        pickable=False
    )


def get_hexagon_layer(data: pd.DataFrame,
                      radius: int = 10000,
                      elevation_scale: int = 10,
                      coverage: float = 0.9) -> pdk.Layer:
    """
    Create a hexagon layer for density visualization.
    
    Args:
        data: DataFrame with lat, lng columns
        radius: Hexagon radius in meters
        elevation_scale: Scale factor for hexagon height
        coverage: Coverage factor (0-1)
    
    Returns:
        PyDeck HexagonLayer
    """
    return pdk.Layer(
        'HexagonLayer',
        data=data,
        get_position=['lng', 'lat'],
        radius=radius,
        elevation_scale=elevation_scale,
        elevation_range=[0, 1000],
        coverage=coverage,
        pickable=True,
        extruded=True
    )


def get_heatmap_layer(data: pd.DataFrame,
                      radius: int = 30,
                      opacity: float = 0.6,
                      intensity: float = 1.0) -> pdk.Layer:
    """
    Create a heatmap layer for order intensity visualization.
    
    Args:
        data: DataFrame with lat, lng columns
        radius: Heatmap radius in pixels
        opacity: Layer opacity
        intensity: Heat intensity
    
    Returns:
        PyDeck HeatmapLayer
    """
    return pdk.Layer(
        'HeatmapLayer',
        data=data,
        get_position=['lng', 'lat'],
        radius_pixels=radius,
        opacity=opacity,
        intensity=intensity,
        threshold=0.05
    )


def get_view_state(lat: float = -15.78,
                   lng: float = -47.93,
                   zoom: float = 3.5,
                   pitch: int = 40,
                   bearing: int = 0) -> pdk.ViewState:
    """
    Create default view state for Brazilian map.
    
    Args:
        lat: Center latitude
        lng: Center longitude
        zoom: Zoom level (2-6 recommended for Brazil)
        pitch: Camera pitch angle (0-60)
        bearing: Camera bearing angle
    
    Returns:
        PyDeck ViewState
    """
    return pdk.ViewState(
        latitude=lat,
        longitude=lng,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing
    )


def get_deck(map_style: str = 'mapbox://styles/mapbox/dark-v11',
             initial_view_state: Optional[pdk.ViewState] = None,
             layers: Optional[List[pdk.Layer]] = None,
             tooltip: Optional[Dict] = None) -> pdk.Deck:
    """
    Create a complete PyDeck deck with configuration.
    
    Args:
        map_style: Mapbox style URL
        initial_view_state: ViewState object
        layers: List of PyDeck layers
        tooltip: Tooltip configuration
    
    Returns:
        PyDeck Deck object
    """
    if initial_view_state is None:
        initial_view_state = get_view_state()
    
    if tooltip is None:
        tooltip = {
            "html": "<b>Route: {seller_state} -> {customer_state}</b><br>Orders: {order_count}<br>Delivery: {avg_delivery_days:.1f} days",
            "style": {"backgroundColor": "black", "color": "white"}
        }
    
    return pdk.Deck(
        map_style=map_style,
        initial_view_state=initial_view_state,
        layers=layers or [],
        tooltip=tooltip
    )


def create_animated_arc_layer(data: pd.DataFrame,
                              frame: int = 0,
                              max_frames: int = 60,
                              dash_array: Optional[List[int]] = None) -> pdk.Layer:
    """
    Create an animated arc layer with moving dash effect.
    
    Args:
        data: DataFrame with route data
        frame: Current animation frame (0 to max_frames-1)
        max_frames: Total number of frames in animation cycle
        dash_array: Dash pattern [dash_length, gap_length]
    
    Returns:
        PyDeck ArcLayer with animation
    """
    if dash_array is None:
        dash_array = [4, 8]
    
    dash_offset = (frame / max_frames) * 100
    
    return get_arc_layer(
        data=data,
        get_width='line_width',
        get_color='color',
        dash_array=dash_array,
        dash_offset=dash_offset
    )


def get_enhanced_scatter_layer(data: pd.DataFrame,
                               size_column: Optional[str] = None,
                               color_column: Optional[str] = None,
                               color_map: Optional[Dict] = None) -> pdk.Layer:
    """
    Create scatter layer with dynamic sizing and coloring.
    
    Args:
        data: DataFrame with lat, lng columns
        size_column: Column name for point size
        color_column: Column name for color category
        color_map: Dictionary mapping categories to RGBA colors
    
    Returns:
        PyDeck ScatterplotLayer
    """
    layer_data = data.copy()
    
    # Handle sizing
    if size_column and size_column in layer_data.columns:
        radius_scale = 50000 / layer_data[size_column].max()
        layer_data['dynamic_radius'] = layer_data[size_column] * radius_scale
        layer_data['dynamic_radius'] = layer_data['dynamic_radius'].clip(5000, 50000)
        radius_field = 'dynamic_radius'
    else:
        radius_field = 15000
    
    # Handle coloring
    if color_column and color_column in layer_data.columns and color_map:
        layer_data['dynamic_color'] = layer_data[color_column].map(color_map)
        color_field = 'dynamic_color'
    else:
        color_field = [50, 50, 80, 120]
    
    return pdk.Layer(
        'ScatterplotLayer',
        data=layer_data,
        get_position=['lng', 'lat'],
        get_radius=radius_field,
        get_fill_color=color_field,
        get_line_color=[255, 255, 255, 80],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True
    )


def create_composite_layer(data: pd.DataFrame,
                           layer_type: str = 'arc',
                           **kwargs) -> pdk.Layer:
    """
    Factory function to create different types of layers.
    
    Args:
        data: DataFrame with required columns
        layer_type: Type of layer ('arc', 'scatter', 'warehouse', 'text', 'hexagon', 'heatmap')
        **kwargs: Additional arguments for specific layer types
    
    Returns:
        PyDeck Layer
    """
    layer_map = {
        'arc': get_arc_layer,
        'scatter': get_scatter_layer,
        'warehouse': get_warehouse_layer,
        'text': get_text_layer,
        'hexagon': get_hexagon_layer,
        'heatmap': get_heatmap_layer
    }
    
    if layer_type not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_type}. Available: {list(layer_map.keys())}")
    
    return layer_map[layer_type](data, **kwargs)


if __name__ == "__main__":
    print("PyDeck layers module loaded")
    print("Available layer functions:")
    print("  - get_arc_layer()")
    print("  - get_scatter_layer()")
    print("  - get_warehouse_layer()")
    print("  - get_text_layer()")
    print("  - get_hexagon_layer()")
    print("  - get_heatmap_layer()")
    print("  - create_animated_arc_layer()")