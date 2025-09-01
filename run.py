# run.py
# Programmatically start streamlit for PyInstaller packaging
from pathlib import Path
import streamlit.web.bootstrap as bootstrap

APP = str(Path(__file__).with_name("app.py"))
bootstrap.run(APP, f'run.py {APP}', [], {})
