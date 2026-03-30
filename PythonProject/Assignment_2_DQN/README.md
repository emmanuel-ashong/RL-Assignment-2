Assignment 2: Deep Q-Learning (DQN) - CartPole-v1

This repository contains the implementation of a Deep Q-Learning agent for the CartPole-v1 environment. It includes a full ablation study evaluating the impact of Target Networks (TN) and Experience Replay (ER).

1.Setup Instructions

To install the necessary dependencies on a university Linux machine, run:

python -m pip install -r requirements.txt


2.Running Experiments

Per the assignment requirements, each configuration is run for $10^6$ steps with five repetitions. Use the following commands to execute each configuration:

Naive DQN:

python main.py --mode naive


Only Target Network (TN):

python main.py --mode only_tn


Only Experience Replay (ER):

python main.py --mode only_er


Full DQN (TN & ER):

python main.py --mode full


3.Generating Results

Once all data collection is complete, generate the required comparison graph (Task 2.4) by running:

python plotting.py


This produces dqn_ablation_results.png, showing the mean and standard deviation for all four configurations.

4.Project Structure

main.py: Main entry point for training.

models.py: Neural Network architecture.

plotting.py: Data aggregation and plotting script.

requirements.txt: Environment configuration.