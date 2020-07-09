import torch
import torch.nn.functional as F
from torch.nn import Linear
from message_passing import MessagePassing
import mw_logging
import logging


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, normalize=False, bias=True,
                 **kwargs):
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""

        logging.debug("---------- forward ----------")

        if torch.is_tensor(x):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        mw_logging.log_tensor(out, "propagate")
        mw_logging.log_peak_increase("propagate")
        rel = self.lin_rel(out)
        mw_logging.log_tensor(rel, "lin_rel")
        mw_logging.log_peak_increase("lin_rel")
        root = self.lin_root(x[1])
        mw_logging.log_tensor(root, "root")
        mw_logging.log_peak_increase("root")
        out = rel + root
        mw_logging.log_tensor(out, "rel + root")
        mw_logging.log_peak_increase("rel + root")

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        mw_logging.log_tensor(out, "Layer out")
        logging.debug("---------- end of forward ----------")

        return out

    def message(self, x_j, edge_weight):
        logging.debug("---------- message ----------")
        mw_logging.log_tensor(x_j, "x_j")

        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
