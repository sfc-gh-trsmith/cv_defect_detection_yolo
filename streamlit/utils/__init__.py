# PCB Defect Detection Dashboard - Utilities

import os
import streamlit as st


def render_svg(svg_path: str, width: int = None, caption: str = None) -> None:
    """
    Render an SVG file inline using st.markdown.
    
    Streamlit in Snowflake doesn't support st.image() with file paths.
    This function reads the SVG content and embeds it directly in HTML.
    
    Args:
        svg_path: Path to the SVG file (relative to the streamlit app root)
        width: Optional width in pixels (default: uses SVG's native width)
        caption: Optional caption to display below the image
    """
    # Get the directory of the streamlit app (one level up from utils)
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(app_dir, svg_path)
    
    try:
        with open(full_path, 'r') as f:
            svg_content = f.read()
        
        # Build inline style
        style = ""
        if width:
            style = f'style="width: {width}px;"'
        
        # Wrap SVG in a div for sizing control
        html = f'<div {style}>{svg_content}</div>'
        
        if caption:
            html += f'<p style="text-align: center; color: #94a3b8; font-size: 0.875rem; margin-top: 0.5rem;">{caption}</p>'
        
        st.markdown(html, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.warning(f"SVG file not found: {svg_path}")
    except Exception as e:
        st.error(f"Error loading SVG: {e}")
