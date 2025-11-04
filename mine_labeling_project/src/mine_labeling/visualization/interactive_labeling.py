"""
Interactive Bounding Box Labeling Tool

Interactive tool for drawing bounding boxes on BMP images
using matplotlib mouse events.

Author: Mine Detection Team
Date: 2025-11-04
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton
from PIL import Image
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class InteractiveBBoxLabeler:
    """
    Interactive bounding box labeling tool

    Controls:
    - Left click + drag: Draw bounding box
    - Right click: Delete nearest bounding box
    - 's' key: Save annotations (XML + JSON)
    - 'q' key: Quit without saving
    - 'Esc' key: Cancel current box
    """

    def __init__(self, image_path: str, output_dir: Optional[str] = None):
        """
        Initialize labeler

        Args:
            image_path: Path to BMP image file
            output_dir: Directory to save annotations (default: same as image)
        """
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir) if output_dir else self.image_path.parent

        # Load image
        self.image = np.array(Image.open(self.image_path))
        self.height, self.width = self.image.shape[:2]

        # Bounding boxes storage
        self.bboxes = []  # List of {xmin, ymin, xmax, ymax}

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_bbox = None
        self.current_rect = None

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.ax.imshow(self.image)
        self.ax.set_title(f'Interactive Labeling: {self.image_path.name}\\n'
                         f'Left-drag: Draw | Right-click: Delete | S: Save | Q: Quit')

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        print("="*60)
        print("Interactive Bounding Box Labeling Tool")
        print("="*60)
        print(f"Image: {self.image_path.name}")
        print(f"Size: {self.width} × {self.height}")
        print()
        print("Controls:")
        print("  - Left click + drag: Draw bounding box")
        print("  - Right click: Delete nearest bounding box")
        print("  - 's' key: Save annotations")
        print("  - 'q' key: Quit without saving")
        print("  - 'Esc' key: Cancel current box")
        print("="*60)

    def on_press(self, event):
        """Handle mouse button press"""
        if event.inaxes != self.ax:
            return

        if event.button == MouseButton.LEFT:
            # Start drawing bounding box
            self.drawing = True
            self.start_point = (event.xdata, event.ydata)

        elif event.button == MouseButton.RIGHT:
            # Delete nearest bounding box
            self.delete_nearest_bbox(event.xdata, event.ydata)

    def on_release(self, event):
        """Handle mouse button release"""
        if not self.drawing:
            return

        if event.button == MouseButton.LEFT and event.inaxes == self.ax:
            # Finish drawing bounding box
            end_point = (event.xdata, event.ydata)

            # Create bbox
            x1, y1 = self.start_point
            x2, y2 = end_point

            xmin = int(min(x1, x2))
            ymin = int(min(y1, y2))
            xmax = int(max(x1, x2))
            ymax = int(max(y1, y2))

            # Validate bbox
            if xmax - xmin > 5 and ymax - ymin > 5:  # Minimum size
                bbox = {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'width': xmax - xmin,
                    'height': ymax - ymin
                }

                self.bboxes.append(bbox)
                self.draw_bbox(bbox, color='red')

                print(f"✓ Added bbox {len(self.bboxes)}: "
                      f"({xmin}, {ymin}) → ({xmax}, {ymax})")

            # Clear current drawing
            if self.current_rect:
                self.current_rect.remove()
                self.current_rect = None

        self.drawing = False
        self.start_point = None

    def on_motion(self, event):
        """Handle mouse motion"""
        if not self.drawing or event.inaxes != self.ax:
            return

        # Update current rectangle
        if self.current_rect:
            self.current_rect.remove()

        x1, y1 = self.start_point
        x2, y2 = event.xdata, event.ydata

        width = x2 - x1
        height = y2 - y1

        self.current_rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='yellow',
            facecolor='none', linestyle='--'
        )

        self.ax.add_patch(self.current_rect)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """Handle key press"""
        if event.key == 's':
            # Save annotations
            self.save_annotations()

        elif event.key == 'q':
            # Quit without saving
            print("Quit without saving")
            plt.close(self.fig)

        elif event.key == 'escape':
            # Cancel current drawing
            if self.current_rect:
                self.current_rect.remove()
                self.current_rect = None
                self.drawing = False
                self.start_point = None
                self.fig.canvas.draw_idle()

    def draw_bbox(self, bbox: Dict[str, int], color: str = 'red'):
        """Draw bounding box on image"""
        rect = patches.Rectangle(
            (bbox['xmin'], bbox['ymin']),
            bbox['width'], bbox['height'],
            linewidth=2, edgecolor=color,
            facecolor='none'
        )

        self.ax.add_patch(rect)
        self.fig.canvas.draw_idle()

    def delete_nearest_bbox(self, x: float, y: float):
        """Delete nearest bounding box to click point"""
        if not self.bboxes:
            return

        # Find nearest bbox
        min_dist = float('inf')
        nearest_idx = None

        for i, bbox in enumerate(self.bboxes):
            # Calculate distance to bbox center
            center_x = (bbox['xmin'] + bbox['xmax']) / 2
            center_y = (bbox['ymin'] + bbox['ymax']) / 2

            dist = ((x - center_x)**2 + (y - center_y)**2)**0.5

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Delete bbox
        if nearest_idx is not None:
            deleted = self.bboxes.pop(nearest_idx)
            print(f"✗ Deleted bbox {nearest_idx+1}: "
                  f"({deleted['xmin']}, {deleted['ymin']}) → "
                  f"({deleted['xmax']}, {deleted['ymax']})")

            # Redraw all boxes
            self.redraw_all_bboxes()

    def redraw_all_bboxes(self):
        """Redraw all bounding boxes"""
        # Clear patches
        for patch in self.ax.patches:
            patch.remove()

        # Redraw all
        for bbox in self.bboxes:
            self.draw_bbox(bbox, color='red')

        self.fig.canvas.draw_idle()

    def save_annotations(self):
        """Save annotations to XML and JSON"""
        if not self.bboxes:
            print("No bounding boxes to save!")
            return

        # Save XML (PASCAL VOC format)
        xml_path = self.output_dir / f'{self.image_path.stem}.xml'
        self.save_xml(xml_path)

        # Save JSON
        json_path = self.output_dir / f'{self.image_path.stem}.json'
        self.save_json(json_path)

        print(f"\\n✓ Annotations saved:")
        print(f"  - XML: {xml_path.name}")
        print(f"  - JSON: {json_path.name}")
        print(f"  - Total boxes: {len(self.bboxes)}")

    def save_xml(self, output_path: Path):
        """Save annotations in PASCAL VOC XML format"""
        root = ET.Element('annotation')

        # Folder
        folder = ET.SubElement(root, 'folder')
        folder.text = str(self.image_path.parent.name)

        # Filename
        filename = ET.SubElement(root, 'filename')
        filename.text = self.image_path.name

        # Path
        path = ET.SubElement(root, 'path')
        path.text = str(self.image_path.absolute())

        # Source
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Manual Labeling'

        # Size
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(self.width)
        height = ET.SubElement(size, 'height')
        height.text = str(self.height)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(self.image.shape[2] if len(self.image.shape) == 3 else 1)

        # Segmented
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'

        # Objects
        for bbox in self.bboxes:
            obj = ET.SubElement(root, 'object')

            name = ET.SubElement(obj, 'name')
            name.text = 'mine'

            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'

            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'

            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(bbox['xmin'])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(bbox['ymin'])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(bbox['xmax'])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(bbox['ymax'])

        # Pretty print and save
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='\\t')
        xml_lines = [line for line in xml_str.split('\\n') if line.strip()]
        xml_str = '\\n'.join(xml_lines)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

    def save_json(self, output_path: Path):
        """Save annotations in JSON format"""
        data = {
            'image': {
                'filename': self.image_path.name,
                'width': self.width,
                'height': self.height,
                'path': str(self.image_path.absolute())
            },
            'annotations': self.bboxes
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def show(self):
        """Show interactive labeling interface"""
        plt.show()


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python interactive_labeling.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    labeler = InteractiveBBoxLabeler(image_path, output_dir)
    labeler.show()


if __name__ == '__main__':
    main()
