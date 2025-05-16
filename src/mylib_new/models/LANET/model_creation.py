from .lanet.LANET import TransformerEncoder



def create_model(opt):
    model = TransformerEncoder(
        num_types=opt.num_types,
        emb_dim=opt.emb_dim,
        nhead=opt.nhead,
        dropout=opt.dropout,
        activation=opt.activation,
        batch_first=opt.batch_first,
        encoder_num_layers=opt.encoder_num_layers,
        layer_norm_eps=opt.layer_norm_eps,
        bias=opt.bias
    )
    return model