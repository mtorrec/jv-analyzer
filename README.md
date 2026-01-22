# JV Curve Analyzer

A Streamlit app for analyzing current-voltage (J-V) characteristics from solar cell measurements.

## Features

- Import IV data files (.txt) with voltage and current columns
- Analyze forward and backward scans for hysteresis measurement
- Calculate key parameters: Jsc, Voc, FF, PCE, Vmp, Jmp, Rs, and hysteresis index
- Organize samples into groups with configurable active areas
- Interactive J-V and P-V plots
- Statistical comparison across groups
- Scan rate analysis: assign scan rates to files and plot parameters vs scan rate
- Export results to CSV or Excel

## Installation

```bash
pip install -r requirements.txt
```

Note: Requires [NREL iv_params](https://github.com/NREL/iv_params):
```bash
pip install git+https://github.com/NREL/iv_params.git
```

## Usage

```bash
streamlit run app.py
```

## Data Format

Input files should contain two columns (space-separated):
1. Voltage (V)
2. Current (mA)

Data should include backward scan followed by forward scan.
