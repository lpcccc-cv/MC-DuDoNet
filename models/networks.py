import models.modules.DuDoNet_Loupe as DuDoNet_Loupe

####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'DuDoNet_Loupe':
        netG = DuDoNet_Loupe.DuDoNet_Loupe(opt_net)

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
