# Using BPTF on a combined ICEWS-GDELT-TERRIER dataset
## Table of contents
- [Description](#description)
- [Cloning](#Cloning_repo)

## Description
Using Bayesian Poisson Tensor Factorisation to decompose dyadic data tensors to analyse country-country behaviours over time and database representation.
The main script is comparing_3_datasets.ipynb.

## Cloning_repo
### Cloning both main and submodules at once
Run 
```bash 
git clone --recurse-submodules https://github.com/yu-sen-lui/bptf_new.git
```
### Cloning main and then submodules separately
Clone the main branch first
```bash
git clone https://github.com/yu-sen-lui/bptf_new.git
```
Then clone the submodules
```bash
cd bptf_new
git submodule update --init -recursive
```
### To update modules
```bash
git submodule update --remote --recursive
```
