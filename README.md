# Serval

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Serval--Decode-blue?logo=github)](https://github.com/Roth-Lab/serval)

## Overview

**Serval** is a fully original decoding pipeline for multiplexed spatial transcriptomics developed by Jenkin Tsui at the University of British Columbia and BC Cancer Research Institute.  
**Serval** is a modular framework that allows users to decode MERFISH images using various decoding methods, including [MERlin](https://github.com/emanuega/MERlin) (Emanuel _et al._, 2020), Simple Nearest Neighbor, and Cosine-optimized decoding.

## Features

- High‐accuracy barcode decoding via cosine‐optimized pixel matching  
- Modular design: easily swap in/out different chromatic correction and filtering steps  

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/Roth-Lab/serval.git
   cd Serval-Decode
   ```

2. Create and activate your conda environment  
   ```bash
   conda create --name serval-dev python=3.9
   conda activate serval-dev
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

_To decode a dataset with default settings:_

```bash
python /examples/standard_run.py \
  /examples/config.json
```

## Datasets

4T1 cell culture dataset: https://doi.org/10.5281/zenodo.15678420
4T1 tissue dataset: https://doi.org/10.5281/zenodo.15678603


## Citation

If you use Serval Decode in your work, please cite:

> Tsui, J. “Serval: A Cosine‐Optimized Pixel Decoder for Spatial Transcriptomics.” *In preparation*, 2025.

## License

Serval Decode is licensed under the modified Apache License, Version 2.0; see the LICENSE file for details.


```text
Copyright © 2025  
Jenkin Tsui. All rights reserved.
```

## Trademarks

All other trademarks referenced herein are the property of their respective owners.
