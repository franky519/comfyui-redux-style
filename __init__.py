from .grid_visualizer import StyleModelGridVisualizer
from .style_model import StyleModelConditioner
from .advancd import StyleModelAdvanced

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "StyleModelGridVisualizer": StyleModelGridVisualizer,
    "StyleModelConditioner": StyleModelConditioner,
    "StyleModelAdvanced": StyleModelAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelGridVisualizer": "🔍 Style Model Grid",
    "StyleModelConditioner": "🎨 Style Model Apply",
    "StyleModelAdvanced": "🎨 Style Model Advanced",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
