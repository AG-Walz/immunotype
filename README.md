# immunotype
Peptide-based HLA typing using immunopeptidomics data

## Install conda environment
```
conda env create -f environment.yml
```

## Activate conda environment
```
conda activate pbtype
```

## Run pbType
```
python immunotype.py test_peptides.tsv --output_file example_prediction.tsv
```

## Example output
```
Predicted HLA typing:
HLA-A01:01, HLA-A01:25, HLA-B15:17, HLA-B40:01, HLA-C16:01
```
