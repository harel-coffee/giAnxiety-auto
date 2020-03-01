# This script converts all plot files in .pdf format to .png
# Paul A. Bloom
# January 2020

# Loop through all plots and make png version
for plot in ../plots/*
  do
    sips -s format png $plot --out ${plot%.pdf}.png
  done

# move png plots to their own subfolder
mkdir ../plots/png
mv ../plots/*.png ../plots/png/