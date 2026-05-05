# TML-Assignment1

# Task 1: Membership Inference Attack

## Description
This project implements a **loss-based reference attack (LRA)** to infer whether a sample was part of a model’s training set.

## Setup
Install dependencies:
pip install -r requirements.txt

## Data & Model
Place the following files in the project directory:
- model.pt (target model)
- pub.pt (public dataset)
- priv.pt (private dataset)

## Execution
Run the attack:
python attack.py

## Output
- submission.csv: contains id and membership score

## Method
A reference model is trained on **non-member public data**.  
Membership scores are computed as:
score = loss_ref − loss_target

Scores are normalized to [0,1].

## Reproducibility
Running the script reproduces the submitted results and generates the final CSV for evaluation.
