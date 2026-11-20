"""Adapt an imported Gaussian PPO action head when its physical scale changes."""
import math
import torch

def rescale_actor(model,old_scale,new_scale):
    policy=model.policy
    if getattr(policy,'squash_output',False) or not isinstance(policy.action_net,torch.nn.Linear) or not hasattr(policy,'log_std'):
        raise ValueError('Automatic travel migration requires the standard unsquashed Gaussian PPO policy')
    ratio=old_scale/new_scale
    with torch.no_grad():
        policy.action_net.weight.mul_(ratio);policy.action_net.bias.mul_(ratio);policy.log_std.add_(math.log(ratio))
    # Adam moments are in the old action coordinates. Reset only the changed tensors.
    for param in [*policy.action_net.parameters(),policy.log_std]:policy.optimizer.state.pop(param,None)
