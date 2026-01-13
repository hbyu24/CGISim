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

This folder stores all experimental outputs and is organized into two main subdirectories: `Simulation-Log` and `Micro-Interactions`.

#### 1. Simulation-Log

This directory contains the raw simulation data, recording the complete dialogue history and decision-making processes for each experiment.

- **Structure**:
  - `MBTI/` & `OCEAN/`: Top-level folders for each personality framework.
  - Inside each framework folder, results are categorized into:
    - `baseline/`: Control group experiments.
    - `Homogeneous/`: Experiments with agents sharing the same personality traits.
    - `Heterogeneous/`: Experiments with diverse personality compositions.

- **Content**:
  Each experiment folder (e.g., `baseline1`, `ENFJ`) contains detailed logs organized as follows:
  - `documents/`: Stores the global simulation logs (`full_record.html/json`) and individual agent subfolders. Each agent's folder contains their `context.json` (state) and `memory.json` (history).
  - `meetings/`: Contains subfolders for specific meeting events (e.g., `Annual_Budget_Year_1`). Inside, you will find `meeting_record.html/json` for the meeting transcript and agent-specific context snapshots.
  - `decision_tracker_data_*.json`: A JSON log file specifically tracking the decisions and outcomes of the simulation.

#### 2. Micro-Interactions

This directory contains the results of micro-level analyses performed on the conversation logs from `Simulation-Log`. These files capture specific interaction metrics and decision dynamics.

- **Structure**:
  - Mirroring the `Simulation-Log` structure, it is divided into `MBTI/` and `OCEAN/`, and further into `baseline/`, `Homogeneous/`, and `Heterogeneous/`.

- **Content**:
  - JSON log files (e.g., `decision_tracker_data_*.json`) containing detailed analysis of specific interaction types and decision quality metrics.