# Estimating Skill Prices in a Model of Continuous Task Choice
*Supplementary repository for my Master's Thesis*
*******************************************************************************

As part of my Master's Thesis I present a Model of continuous task choice that can be used in estimations of skill prices. Besides the identification of this model, I make use of a simuation study to illustrate the application of my results.
This repository contains my thesis as well as the program code that I use for the simulation study.

My thesis can be produced by building the *main.tex* file in the *latex_files* subdirdectory. The results of the simulation study are created and presented in a jupyter notebook (*simulation.ipynb*).

The filestructure of this repo is as follows:

**MC_simulation**: This directory contains all codes used on the simulation study as well as the resulting outputs and plots.
* **DPG**: Contains the python code used for the data generating process.
.- mc_skills.py: Functions used in the simulation of task specific skill endowments of workers.
.- mc_prices.py: Contains functions for the simuation of task specific skill prices.
.- mc_optimal_wage_choice.py: Relies on simulated skill prices and endownments and provides optimal task choices and resulting utilities and wages.
.- draw_data.py: Wraps around the DGP and returns the simulation data
* **FIG**: Figures that are produced in the simulation study and are used in my thesis are stored here.
* **OUT**: In case one decides to store the simulation data as json files, it will be stored in this folder.
* **old_code**: Deprecated python codes that are not used in the current state of the project.

**latex_files**: Contains all LaTex files that build my thesis.
* **Appendix**: Various subsections of the appendix.
* **Sections**: Each section of my thesis is contained in the respective files in this directory.
* *main.tex*: Main document of the thesis. This file imports all other subfiles and builds my thesis.
