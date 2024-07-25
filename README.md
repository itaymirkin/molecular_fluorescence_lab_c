# molecular_fluorescence_lab_c
molecular fluorescence repo 
# MATLAB Data Analysis Scripts

This repository contains two MATLAB scripts for analyzing spectroscopic and image data. These scripts are designed for processing and visualizing experimental results in materials science or chemistry research.

## Scripts

1. `processSheet_code.m`: Analyzes spectroscopic data from Excel files.
2. `image_fluorescence.m`: Processes and analyzes a series of images, extracting intensity profiles and performing linear fits.

## Spectroscopic Analysis Script

### Features:
- Reads data from multiple sheets in an Excel file
- Processes spectroscopic data for different concentrations
- Calculates integrated intensities and normalizes them
- Generates plots for normalized integrated intensity vs. concentration
- Performs linear fits for small concentrations
- Calculates molar attenuation coefficients

### Usage:
1. Update the `filename` variable with your Excel file name.
2. Adjust the `dataColumns`, `timeColumns`, and `concentrations` arrays if necessary.
3. Run the script in MATLAB.

## Image Analysis Script

### Features:
- Processes a series of images (Fe1.jpeg to Fe10.jpeg)
- Allows user to select a line of interest on each image
- Extracts intensity profiles along the selected lines
- Performs logarithmic transformations and linear fits
- Generates various plots for each image:
  - Selected line on the image
  - Intensity profile
  - Log-transformed intensity with linear fit
  - Residuals plot

### Usage:
1. Ensure your images (Fe1.jpeg to Fe10.jpeg) are in the same directory as the script.
2. Run the script in MATLAB.
3. For each image, if it's the first time:
   - Select two points to define the line of interest when prompted.
   - Enter the width (in pixels) for averaging intensity.
4. The script will generate and save multiple plots for each image.

## Requirements

- MATLAB (version R2019b or later recommended)
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

## Output

Both scripts generate multiple figures and save them as PNG files in the current directory. The spectroscopic analysis script also outputs results to the MATLAB console.

## Notes

- These scripts are designed for specific experimental setups and may need modifications to work with different data formats or experimental conditions.
- Ensure you have the necessary permissions to read/write files in the directory where you're running these scripts.

## Contributing

Feel free to fork this repository and submit pull requests with improvements or bug fixes. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](https://choosealicense.com/licenses/mit/)
