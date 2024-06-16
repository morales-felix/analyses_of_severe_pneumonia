# Analyses of severe pneumonia patients

This repository contains Jupyter notebooks summarizing work done for my Quantitative and Systems Biology masters thesis research project. *As such, it is an example of my first body of work as a Data Scientist*.  

# Main question (tl;dr)  
*Among the sea of tests ordered by clinicians, which ones would be the most informative on discharges of severly-ill pneumonia patients? And can we build a model that predicts discrete patient states, and the transitions between these states?*

# Verbose description
This work was done within the Successful Clinical Response in Pneumonia Treatment (SCRIPT), which by January 2022 (the time the analysis was done) had enrolled ~600 critically-ill patients, on suspicion of pneumonia. The overall goal of this effort is to find out what drives poor outcomes and treatment responses in these patients. To pin down these mechanisms at the cellular/gene level, this center had the unique advantage of having developed a safe and repeatable protocol to collect Bronchoalveolar Lavage samples from these patients: In essence, we can sample fluid from their lungs, and take a look at the cellular composition of these samples. In this way, the center aims to connect the dots between the clinical presentation of these patients, and the biological mechanisms driving these clinical courses.  

My work within this project focused in wrangling the clinical data present in the EHR to find what is informative on patient outcomes, and make it amenable to various modeling schemes. That work led to exploring modeling approaches aimed to characterize patient types/states within the ICU, which resulted in a manuscript submitted to the Proceedings of the National Academy of Sciences (PNAS). With robustly-identified clinical states, we'd like to build a model predictive of transitions between states, in the hopes that such predictions inform about the underlying biology driving changes in the clinical course of patients.

# How to read this repo  
This repo mainly consists of Jupyter notebooks created for analysis. They are numbered roughly in the order they were created, so they tell a story of continuous discovery of patterns (or issues with the data). Read them in whatever order you perfer, but going in numerical order might be best.
