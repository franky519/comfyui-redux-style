from .grid_visualizer import StyleModelGridVisualizer
from .style_model import StyleModelConditioner

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "StyleModelGridVisualizer": StyleModelGridVisualizer,
    "StyleModelConditioner": StyleModelConditioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelGridVisualizer": "🔍 Style Model Grid",
    "StyleModelConditioner": "🎨 Style Model Apply",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
