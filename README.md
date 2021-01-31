# Particle Flow alternatives using Machine Learning

**Checkout the package** <br/>
git clone https://github.com/sanmayphy/ENERGYFLOW_ML.git

The tasks are defined at __https://docs.google.com/document/d/1DnOqVxdA-5WjEkjYZtV8TW40YKIukPSwsyMnizDWnbE/edit__

* DATA_PROCESSING

for Pflow production first you need to run topocluster.

1. python readTree_Topo.py

Open the readTree_Topo.py to choose if you wish to run over charged,
neutral or total energy.  This produced a file with the Topolcustr information.

Open perCellPflow.py and change the input file.
Please note that a file with topoclusterng inputs exist in eos already: /afs/cern.ch/work/f/fdibello/public/Outfile_2to5GeV_TotalTopo.h5

2. run python perCellPflow.py

This produces a file with Pflow predition.
Pflow prediction are also available on eos as well as the Topoclustering file
from script 1. No need to re-run these if only the NN has changed.

this is the Pflow prediction: /afs/cern.ch/work/f/fdibello/public/Outfile_2to5GeV_TotalEpred.h5

3.   python Evaluate_Allpy
Edit the macro to get the correct inputs link.
This is for evaluation. Add your new network to the list and this will output
the  figure of merit used to quantify the performance. Feel free to add
additional plots if you like.

* PUBLIC DATASET
A part of the dataset for PFlow and Superresolution is available through : https://zenodo.org/record/4483330#.YBaxYHczZBw
