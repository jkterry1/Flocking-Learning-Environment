#!/bin/bash

rsync -avr --progress --stats gcp-flocking-1:~/rl-baselines3-zoo-flocking/results/ ./Logs/
# rsync -avr --progress --stats dream:~/rl-baselines3-zoo-flocking/results/ ./Logs/
# rsync -avr --progress --stats prophet:~/rl-baselines3-zoo-flocking/results/ ./Logs/
