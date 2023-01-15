## SarmaHybrid

# General description
This repository aims to implement a scientific machine learning model in a scenario where only parts of the system (in terms of variables and variable interactions) are known. The test case chosen fr thsi approach is a MAPK cascade model by [Sarma and Ghosh (2012)](http://www.biomedcentral.com/1756-0500/5/287). The repository contains an implementation of their model (currently only S2, S1 to be implemented later) and a hybrid version of the same model where some intermediate species are taken out, see Figure 1 below, which is a slide from a presentation I had given. 

![Figure 1, true and hybrid model structures.](model_comparison.PNG?raw=true "blabla")

The hybrid model contains [augmented neural ODEs](https://dl.acm.org/doi/10.5555/3454287.3454569) which both link the disjoint parts of the model and provide additional dimensionality to represent the missing variables.


