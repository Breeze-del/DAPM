# A pytorch implementation for paper "Decision-Aware Preference Modeling for Multi-Behavior Recommendation" 

## REQUIREMENTS
1. pytorch 1.4
2. loguru
3. scipy
4. sklearn


## EXPERIMENTS

Here we use the GMF as the CF basic model.

How to train Beibei : python search.py 0 [GPU_ID]

How to train Taobao : python search.py 1 [GPU_ID]

Our model will be restored in the checkpoints.
