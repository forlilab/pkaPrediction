# pkaPrediction

This repo holds the code and weights for pkaPrediction models.

# Use pKa prediction.

If you want to use the model (or the rules) to predict the pKa of any
rdkit mol, some helper functions are provided for convenience.

```python
from rdkit import Chem
from predicting.pkaCalc import calculate_pka

mol = Chem.MolFromSmiles("NCCO")

pka = calculate_pka(mol, method = "etr1") # returns PkaResult object with the results of the model

pka = calculate_pka(mol, method = "rules") # returns PkaResult object with the pka from the rules file

```

# History

- ETR: ExtraTrees Regressor model. Features: uses the predefined rules present
  molscrub, including the approximnate pKas, as well as the standard rdkit descriptors.

The dataset used to train the model comes from:
https://github.com/czodrowskilab/Machine-learning-meets-pKa
