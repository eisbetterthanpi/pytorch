
def track_false(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False
# model.apply(track_false)


# @title reset batchnorm
# print(model.state_dict()['_orig_mod.bn1.running_mean'][0])

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

# model.apply(deactivate_batchnorm)

