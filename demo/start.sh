#!/bin/sh
# starts the demo, run it from the project folder: sh demo/start.sh

# make the virtual environment and install the libraries the first time
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
. .venv/bin/activate

# download the images if they are not there yet
if [ ! -d data/EuroSAT_RGB ]; then
  sh data/get_data.sh
fi

# train the models if they are not saved yet
for m in mlp cnn; do
  if [ ! -f models/$m.pt ]; then
    python code/$m.py
  fi
done

# open the browser a few seconds after the server starts
(sleep 3; open http://127.0.0.1:8000) &
python demo/app.py
