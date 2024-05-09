# IntraSeismic

This repository provides reproducible materials for **DW0026 - IntraSeismic: Using Coordinate-Based Learning for Seismic Inversion** by authors Romero J., Heidrich W., Luiken N., and Ravasi M.


## Project Structure
The repository is organized as follows:

* :open_file_folder: **intraseismic**: A Python library that includes routines for dataset management, diferent types of coordinates encoding, the IntraSeismic model, train functions, and plotting functions.
* :open_file_folder: **data**: A folder containing the data or instructions on how to obtain it.
* :open_file_folder: **notebooks**: A collection of Jupyter notebooks that document the application of IntraSesimic to the inversion of Marmousi, SEAM, and Sleipner datasets.

## Notebooks
The provided notebooks include:

* :open_file_folder: **Marmousi**
  - :orange_book: ``Marm_data_creation.ipynb``: Creates post-stack synthetic seismic datasets with varying noise levels for the Marmousi model.
  - :orange_book: ``Poststack_IS_Marm.ipynb``: Demonstrates the inversion of Marmousi seismic data with a noise level of $\sigma = 0.1$ using IntraSeismic.
  - :orange_book: ``Poststack_IS_Marm_MCUQ.ipynb``: Conducts Monte-Carlo Dropout uncertainty quantification in IntraSeismic.
  - :orange_book: ``Prestack_IS_3nets_Marm.ipynb``: Pre-stack seismic inversion of Marmousi model using IntraSeismic.
  
* :open_file_folder: **SEAM**
  - :orange_book: ``SEAM_data_creation.ipynb``: Creates post-stack synthetic seismic datasets with various noise levels for the SEAM model.
  - :orange_book: ``Poststack_IS_SEAM.ipynb``: Shows the inversion of SEAM seismic data with a noise level of $\sigma = 0.1$ using IntraSeismic.

* :open_file_folder: **Sleipner**
  - :orange_book: ``Poststack_IS_Sleipner.ipynb``: Demonstrates the inversion of the Sleipner seismic dataset using IntraSeismic.

## Getting Started :space_invader: :robot:
To reproduce the results, use the `environment.yml` file for environment setup.

Execute the following command:
```
./install_env.sh
```
The installation takes some time. If you see `Done!` in your terminal, the setup is complete.

Finally, run:
```
pip install -e . 
```
in the folder where the setup.py file is located.


Always activate the environment with:
```
conda activate my_env
```

**Disclaimer:** Experiments were conducted on an AMD EPYC 7713 64-Core processor equipped with a single NVIDIA TESLA A100. Different hardware may require alternate environment configurations.


