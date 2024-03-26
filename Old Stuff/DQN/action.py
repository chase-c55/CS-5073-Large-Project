import math
import random
import torch


# NOTE:
# steps_done: This is an artifact of the problem we are looking at
#             we want to adjust how we sample based on the number of
#             moves taken
#
#             Really, we would want to roll this into state as a parameter
#             that we track but don't build our model on.
def select_action(
    state, env, steps_done, policy_net, eps_start, eps_end, eps_decay, device
):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
        -1.0 * steps_done / eps_decay
    )
    # steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).argmax().view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )
