from types import SimpleNamespace

import torch
import pytest

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.losses import _remove_pose, _TARGET_FLOOR
from arm_planner.data_free.surface_diversity import SurfaceDiversity, shell_probes, surface_distances


def test_shell_counts_and_volume_weights():
    pts, w = shell_probes(2048, .9, torch.Generator().manual_seed(160))
    r = pts.norm(dim=-1)
    for low, high, n in [(0,.2,102),(.2,.55,1024),(.55,.7,410),(.7,.9,512)]:
        mask = (r >= low) & (r < high)
        assert mask.sum() == n
        torch.testing.assert_close(w[mask].sum(), torch.tensor((high**3-low**3)/.9**3))


def test_bidirectional_matching_and_quality_budget():
    points = torch.tensor([[[0.,0,0],[1,0,0],[3,0,0],[7,0,0]],
                           [[0.,1,0],[1,1,0],[3,1,0],[17,1,0]]], requires_grad=True)
    near = torch.tensor([[1],[0]])
    cfg = SimpleNamespace(surface_start_queries=2, surface_growth_interval=1)
    fp = torch.zeros(2,4)
    norms = torch.ones_like(fp)
    for step, count in [(0,2),(1,4)]:
        d, info = surface_distances(points,fp,norms,near,cfg,step)
        full = torch.cdist(points[0],points[1])
        expected = .5*(full[:count].min(1).values.mean()+full[:,:count].min(0).values.mean())
        torch.testing.assert_close(d.flatten(),expected.expand(2))
        assert info["surface/query_min"] == count
    d, info = surface_distances(points,fp,norms+.2,near,cfg,1)
    assert info["surface/query_max"] == 2
    d, info = surface_distances(points,fp+.1,norms,near,cfg,1)
    assert d.sum()==0 and info["surface/pair_coverage"]==0


@pytest.mark.parametrize("defer_score", [False, True])
@pytest.mark.parametrize("fixed_direction,move_field", [(False, True), (True, True), (True, False)])
def test_replayed_diversity_matches_direct_autograd(fixed_direction, move_field, defer_score):
    cfg = DataFreeConfig(width=16, layers=3, skip_layer=1, mapping_width=16, latent_dim=4,
                         surface_match_weight=1., surface_sdf_probes=64, surface_start_queries=8,
                         nearest_k=2, field_moves_surface=move_field, variety_weight=1.3,
                         surface_fixed_direction=fixed_direction)
    torch.manual_seed(5991)
    model = ModulatedSphereDecoder(cfg)
    z = torch.nn.functional.normalize(torch.randn(5,4))
    scorer = SurfaceDiversity(model,z,cfg,torch.Generator().manual_seed(258),0,defer_score=defer_score)
    scorer.finish_score()
    assert scorer.stats["surface/point_gradient_norm"] > 0
    scorer.backward()
    replay = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    x,w = shell_probes(64,cfg.bound_radius,torch.Generator().manual_seed(258))
    mod = model.prepare(z)
    f,g = field_and_grad(model,mod,x.expand(len(z),-1,-1))
    p = x-f[...,None]*(g.detach() if fixed_direction else g)
    fp = model.query(mod,p.detach())
    live = f+(fp-fp.detach()) if move_field else f
    raw = torch.cdist(f.detach()*w,f.detach()*w,p=1)
    raw.fill_diagonal_(torch.inf)
    near = raw.topk(2,largest=False).indices
    surface,_ = surface_distances(p,fp,g.norm(dim=-1),near,cfg,0)
    a = torch.arange(len(z))[:,None].expand_as(near).reshape(-1)
    b = near.flatten()
    a_f,b_f=live[a],live[b]
    delta=_remove_pose(a_f-b_f,x.expand(len(a),-1,-1),g[a].detach(),
                       .5*(a_f+b_f).detach(),torch.ones_like(a_f),w)
    distance = (delta.abs()*w).sum(1).reshape_as(near)+surface
    soft = (distance+_TARGET_FLOOR).pow(-cfg.nearest_power).sum(1).pow(-1/cfg.nearest_power)
    loss=-(soft/cfg.diversity_slope+.001).log().mean()
    torch.testing.assert_close(loss,scorer.value)
    (cfg.variety_weight*loss).backward()
    for a,b in zip(replay,[p.grad for p in model.parameters() if p.grad is not None],strict=True):
        torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
