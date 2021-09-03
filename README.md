# sagetrader

AWS Sagemaker project for financial analysis.

## Getting the data

The data is pulled from the Lynx API:

https://api.lynx.academy/?id=lynx-api-documentation

Examples of the API usage:

https://github.com/lynxbroker/API-examples/tree/master/Python

The data is stored in an S3 bucket.
After each data pull, the prediction runs on the most current data,
generating a decision about placing, cancelling or changing the buy/sell orders.

## Reinforcement learning

The project uses RL methods in order to maximize the expected income generated by the model.
The model is built with the Keras library, and the RL environment is built in complience
with the OpenAI Gym library.

The policy is defined as model mapping an environment state into an action.
For finding the optimal policy we use a simple approach of updating the model 
whenever the reward (immediate or total) is sufficiently high (compared to the average reward).

The state of the environment is defined as the sequences of quotes for each company.

The action is defined as the confidence for each company to bring profit, 
the buy price for each company in case we want to buy it, 
and the sell price for each company that is currently in the portfolio.

The reward function is defined as the income relative to the previous step capital.
