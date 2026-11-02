#!/bin/sh
# downloads the EuroSAT RGB images (about 95 MB)
# run it from the final-project folder: sh data/get_data.sh
cd data
curl -L -O https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip
unzip -q -o EuroSAT_RGB.zip
rm EuroSAT_RGB.zip
