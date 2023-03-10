# Evidential Uncertainty Sampling

Experiments for the use of evidential uncertainty sampling.  

## How to use

Modify and execute *main.py*.  

```python
DATASET = "IRIS" 
CERTAINTY = "UNC"
```

DATASET accepted values : IMP, IRIS, LINE, SIN, CIRCLE, LOG, TRIPLE, DOG  
CERTAINTY accepted values : UNC, EP, EV_UNC, PL

Run code:  
*python3 main.py*

## Datasets

The dataset can be chosen from the following:  

LINE | SIN | CIRCLE | IRIS  
:--:|:--:|:--:|:--:
<img src="extra/line.png" width="100"> |  <img src="extra/sin.png" width="100"> | <img src="extra/circle.png" width="100"> | <img src="extra/iris.png" width="100">

LOG | IMP | TRIPLE | DOG  
:--:|:--:|:--:|:--:
<img src="extra/log.png" width="100"> |  <img src="extra/imp.png" width="100"> | <img src="extra/triple.png" width="100"> | <img src="extra/dog.png" width="100">

## Uncertainties

The uncertainty used for sampling can be chosen from the following:  

UNC | EP | EV_UNC | PL
:--|:--:|:--:|:--:
Classical | Epistemic | Klir | Evidential Epistemic

Note that for Epistemic and Evidential Epistemic uncertainties, aleatoric and total uncertainties are also displayed.  
For Klir uncertainty, Discord and Non-specificity are also displayed.

## Example

Example on the dataset Iris with the least confidence measure. The following parameters are used:  

```python
DATASET = "IRIS" 
CERTAINTY = "UNC"
```

Output:  

DATASET | UNCERTAINTY 
:--:|:--:
<img src="extra/iris.png" width="220"> |  <img src="extra/unc.png" width="220"> 

## Faster run

Computation time can be reduced by reducing the size of the uncertainty grid (it will only lower the definition of the representation):

```python

# Size of the uncertainty grid,
# can be reduced to go much faster
SIZE_X1 = 60
SIZE_X2 = 50

```
