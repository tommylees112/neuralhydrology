class LinearHiddenProbe(BaseModel):
    module_parts = ["head"]

    def __init__(self, cfg: Config, n_out: int = 1, dropout: float = 0.0):
        super(LinearHiddenProbe, self).__init__(cfg=cfg)
        self.n_in = cfg.hidden_size
        self.n_out = n_out
        self.dropout = nn.Dropout(p=dropout)

        self.head = get_head(cfg=cfg, n_in=self.n_in, n_out=self.n_out)

    def forward(self, data: Dict[str, Union[Tensor, str]]):
        pred = {}
        pred.update(self.head(self.dropout(data["X"])))

        return pred
