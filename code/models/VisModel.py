import torch
import torch.nn as nn

class VExtractor(nn.Module):
    # Visual feature extractor
    def __init__(self, cfg):
        super(VExtractor, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.v_embd_dim = self.model_cfg.CNN.EMBD_DIM
        self.v_feat_dim = self.model_cfg.CNN.FEAT_DIM        
        self.fc_l = nn.Linear(5, self.v_embd_dim)
        self.fc_dl = nn.Linear(5, self.v_embd_dim)
        self.v_embed = nn.Linear(2 * self.v_feat_dim + self.v_embd_dim, self.v_embd_dim)
        self.v_embed_diff = nn.Linear(self.v_feat_dim + self.v_embd_dim, self.v_embd_dim)
        self.feats_dim = self.v_embd_dim

    def forward(self, visual_feats):
        
        # visual_feats batch x attn_num x [4096*3+10]
        # v_g batch X 4096
        v_g = visual_feats[:, 0, :self.v_feat_dim]
        # v_o batch X 4096
        v_o = visual_feats[:, 0, self.v_feat_dim:2 * self.v_feat_dim]
        # l_o batch X 1024
        l_o = self.fc_l(visual_feats[:, 0, 2 * self.v_feat_dim:(2 * self.v_feat_dim + 5)])
        # v_d batch X attn_num X 4096
        v_d = visual_feats[:, :, (2 * self.v_feat_dim + 5):(3 * self.v_feat_dim + 5)]
        # l_d batch X attn_num X 1024
        l_d = self.fc_dl(visual_feats[:, :, -5:])
        # v_diff_r 
        v = torch.cat([v_g, v_o, torch.relu(l_o)], -1)
        v_r_d = torch.cat([v_d, torch.relu(l_d)], -1)
        v_r = self.v_embed(v)
        v_diff_r = self.v_embed_diff(v_r_d)
        visual_feats = (v_r, v_diff_r)

        return visual_feats


