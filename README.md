# Particle Flow alternatives using Machine Learning

**Checkout the package** <br/>
git clone https://github.com/sanmayphy/ENERGYFLOW_ML.git

The tasks are defined at __https://docs.google.com/document/d/1DnOqVxdA-5WjEkjYZtV8TW40YKIukPSwsyMnizDWnbE/edit__

* DATA_PROCESSING

for Pflow production first you need to run topocluster. python readTree_Topo.py
would work. Open the readTree_Topo.py to choose if you wish to run over charged,
neutral or total energy. As a second step you can run python perCellPflow.py
and then  python Evaluate_TCPos.py .
this macro is very easy and shall be clear what that is doing lokking at it.
Basically makes plot from the predicted energy within the TC.
