# rp-help-tools

# Setup
python3 -m venv .env
source .env/bin/activate
pip install pip-tools
pip-compile --output-file=requirements.txt requirements.in