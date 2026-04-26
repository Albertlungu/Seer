"""
Seer application entry point.

Run from source:  python main.py
Bundled:          open Seer.app  /  Seer.bat  /  ./Seer.sh
"""

from src.seer_app import SeerApp
from src.video_processing.environment import AGGREGATION_PATH

app = SeerApp(aggregation_path=AGGREGATION_PATH, debug=False)
app.run()
