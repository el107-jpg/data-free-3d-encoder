import torch
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.slope_diversity import SlopePairs, _frame

def test_packed_replay_matches_both_live_endpoints():
    torch.manual_seed(71)
    cfg=DataFreeConfig(latent_dim=4,width=16,layers=3,skip_layer=0,mapping_width=12,nearest_k=2,output_init_std=.04)
    model=ModulatedSphereDecoder(cfg)
    codes=torch.nn.functional.normalize(torch.randn(5,4))
    x=torch.nn.functional.normalize(torch.randn(96,3))*cfg.base_radius
    f,g=field_and_grad(model,model.prepare(codes),x.expand(5,-1,-1),create_graph=False)
    pairs=SlopePairs(model,codes,x,f.detach(),g.detach(),cfg)
    assert pairs.mean_sites>0
    weights=torch.randn_like(pairs.table);(pairs.table*weights).sum().backward()
    pairs.backward()
    replay=[p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    _,grad=field_and_grad(model,model.prepare(codes[pairs.ids]),pairs.pair_points,create_graph=True)
    (pairs.angles(grad)*weights[pairs.src,pairs.dst]).sum().backward()
    assert max(float(v.abs().max()) for v in replay)>1.e-5
    for p,v in zip(model.parameters(),replay):torch.testing.assert_close(p.grad,v,atol=1.e-5,rtol=1.e-3)

def test_transformed_correspondence_sign_and_normal_frame():
    torch.manual_seed(12)
    x=torch.randn(512,3)*.3
    # A field with a nondegenerate tangent frame; compare two poses of the same field.
    def value(v):
        a,b,c=v.unbind(-1)
        return .6-a*a-2*b*b-3*c*c+.4*a*b*c+.2*a.sin()
    def slope(v):
        v=v.detach().requires_grad_()
        return torch.autograd.grad(value(v).sum(),v)[0]
    w=torch.tensor(.05);c,s=w.cos(),w.sin()
    rot=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]])
    offset=torch.tensor([.012,-.008,.006])
    moved=x@rot.T+offset
    gb=slope(x);ga=slope(moved)@rot
    R,t,scale=_frame(x,value(moved)[None],value(x)[None],gb[None],torch.ones(len(x),dtype=torch.bool))
    aligned=(x@R[0].T)/scale[0]+t[0]
    corrected=slope(aligned)@R[0]
    angle=lambda a,b: 1-torch.nn.functional.cosine_similarity(a,b).mean()
    assert angle(ga,corrected)<.05*angle(ga,gb)
