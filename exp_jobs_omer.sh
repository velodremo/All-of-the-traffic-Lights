#!/usr/bin/env bash
python3 train_model.py /AI_project/configurations/straight_p0.015_just_delay_agent0.json
python3 train_model.py /AI_project/configurations/straight_p0.015_just_delay_agent0_r_weight0.85.json
python3 train_model.py /AI_project/configurations/straight_p0.015_just_delay_two_agents.json
python3 train_model.py /AI_project/configurations/straight_p0.015_just_delay_two_agents_r_weight0.85.json
python3 train_model.py /AI_project/configurations/straight_p0.015_just_delay_two_agents_r_weight0.85.json
python3 train_model.py /AI_project/configurations/straight_p0.015_just_wait_agent0.json
python3 train_model.py /AI_project/configurations/straight_p0.015_just_wait_agent0_r_weight0.85.json
python3 train_model.py /AI_project/configurations/straight_p0.015_just_wait_two_agents.json
python3 train_model.py /AI_project/configurations/straight_p0.015_just_wait_two_agents_r_weight0.85.json
python3 train_model.py /AI_project/configurations/straight_p0.015_two_agents_lr1e-6.json