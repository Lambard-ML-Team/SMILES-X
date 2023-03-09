# SMILES-X
The **SMILES-X** is an autonomous pipeline that finds best neural architectures to predict a physicochemical property from molecular SMILES only (see [OpenSMILES](http://opensmiles.org/opensmiles.html)). No human-engineered descriptors are needed. The SMILES-X has been especially designed for small datasets (<< 1000 samples). 

Read our open access paper ["SMILES-X: autonomous molecular compounds characterization for small datasets without descriptors"](https://iopscience.iop.org/article/10.1088/2632-2153/ab57f3).

## Who can use the SMILES-X?
Researchers/engineers/students in the fields of materials science, physicochemistry, drug discovery and related fields
 
## Which kind of data can be used?
The SMILES-X is dedicated to **small datasets (<< 1000 samples) of (molecular SMILES, experimental/simulated property)**

## What can I do with it?
With the SMILES-X, you can:
* **Design specific neural architectures** fitted to your small dataset via Bayesian optimization.
* **Predict molecular properties** of a list of SMILES based on designed models ensembling **without human-engineered descriptors**.
* **Interpret a prediction** by visualizing the salient elements and/or substructures most related to a property

## What is the efficiency of the SMILES-X on benchmark datasets?
![table1](/images/Table1_SMILESX_paper.png)

* ESOL: logarithmic aqueous solubility (mols/L) for 1128 organic small molecules.
* FreeSolv: calculated and experimental hydration free energies (kcal/mol) for 642 small neutral molecules in water.
* Lipophilicity: experimental data on octanol/water distribution coefficient (logD at pH 7.4) for 4200 compounds. 

All these datasets are available in the `data/` directory above. 

## Requirements
**SMILES-X works on CPU, but it is highly recommended to have an access to GPU**:</br>
* CUDA == 10.1
* cuDNN == 8.0.3

For now, the SMILES-X has been successfully runned on Titan(Xp, V, V100, P100), GTX 1660 and RTX 2070/80 NVIDIA GPUs.</br>
</br>
For a good start, follow the [RDKit installation guide](https://www.rdkit.org/docs/Install.html) for installing the RDKit via conda.</br>
Then, install the following dependencies in your RDKit conda environment (e.g. my-rdkit-env):</br>
* python == 3.7
* pandas == 1.3.5
* numpy == 1.18.5
* matplotlib == 3.5.2
* GPy == 1.10.0
* GPyOpt == 1.2.6
* scikit-learn == 1.0.2
* adjustText == 0.8
* scipy == 1.7.3
* tensorflow == 2.1.3

## Usage
Please use the python notebook `Example.ipynb` it as a guide. 

## How to cite the SMILES-X?
```
@article{lambard2020smiles,
  title={SMILES-X: autonomous molecular compounds characterization for small datasets without descriptors},
  author={Lambard, Guillaume and Gracheva, Ekaterina},
  journal={Machine Learning: Science and Technology},
  volume={1},
  number={2},
  pages={025004},
  year={2020},
  publisher={IOP Publishing}
}

```
