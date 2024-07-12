import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'joint-optimize':
        from .ref_rec_base_model_loupe import BaseModel as M
    elif model == 'ref-rec':
        from .ref_rec_base_model import BaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
