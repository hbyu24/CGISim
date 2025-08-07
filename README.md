# Personality Diversity as an Antidote to Group Polarization in AI Collective Decision-Making

## Repository Structure

The root directory of this repository contains three main folders: `code`, `data`, and `results`.

### ./data/

This folder contains all the core datasets required to drive our simulation experiments.

We provide data for two personality frameworks:

- **MBTI**: Files such as `mbti_1024_bank.json`, used to generate agents with MBTI personality traits.
- **Big Five (OCEAN)**: Files such as `OCEAN_Data.json` and `NPTI_description.json`, used to generate agents with Big Five personality traits.

### ./code/

This folder contains the source code used in our research, primarily showcasing the implementation for the Big Five (OCEAN) version.

The code covers all experimental conditions, including the project code for baseline, Homogeneous, and Heterogeneous experiments.

> **Note**: The experimental code for the MBTI version is logically identical to the code here; the only difference is the injected prompts and personality descriptions. To avoid redundancy, we have not included the MBTI version of the code.

### ./results/

This folder shows the directory structure for the results of all our experiments.

It contains two subfolders, corresponding to the two personality frameworks: MBTI and OCEAN.

Within both the MBTI and OCEAN folders, there are three subfolders—baseline, Homogeneous, and Heterogeneous—to store the results of the corresponding experiments.