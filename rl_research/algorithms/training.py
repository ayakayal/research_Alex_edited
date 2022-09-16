# def train(env, agent_class, num_iter=100, logs=False):

#     agent = agent_class(env)

#     if logs:
#         agent.train(num_iter)
#     else:
#         agent.train_without_logs(num_iter)

#     return agent
def train(env, agent_class, num_iter=100, logs=False,beta_exp=1):

    agent = agent_class(env,beta_exp)

    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent
