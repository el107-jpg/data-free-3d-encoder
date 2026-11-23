"""Bounded-memory normal comparison for J7's shared, batch-wide neighbours."""
import torch
from torch.nn import functional as F
from .field import field_and_grad

def _frame(x, a, b, gb, inside):
    """Fit A(x) ~ scale B(R x / scale + t), to first order, with J7's pose basis."""
    basis = torch.cat((gb, torch.linalg.cross(x.expand_as(gb), gb),
                       (0.5 * (a+b) - (gb*x).sum(-1))[..., None]), -1)
    basis = basis * inside[None, :, None]
    gram = basis.transpose(1, 2) @ basis
    ridge = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.e-12)
    rhs = basis.transpose(1, 2) @ ((a-b)*inside)[..., None]
    theta = torch.linalg.solve(gram + torch.eye(7,device=x.device)*ridge[:,None,None]*1.e-4, rhs)[..., 0]
    # The linear fit is a local estimate. Limit extrapolation on distant shapes.
    t = theta[:, :3]
    t = t * (.15/t.norm(dim=-1,keepdim=True).clamp_min(1.e-8)).clamp(max=1)
    rot = theta[:, 3:6]
    rot = rot * (.35/rot.norm(dim=-1,keepdim=True).clamp_min(1.e-8)).clamp(max=1)
    a,b,c = rot.unbind(-1)
    zero = torch.zeros_like(a)
    skew = torch.stack((zero,-c,b,c,zero,-a,-b,a,zero),-1).reshape(-1,3,3)
    return torch.matrix_exp(skew),t,theta[:,6].clamp(-.2,.2).exp()

class SlopePairs:
    """Cache values once; replay only the normal-gradient VJP before the optimiser step."""
    @torch.no_grad()
    def __init__(self, model, codes, x, f, g, cfg):
        self.model,self.codes = model,codes.detach()
        n,m=f.shape
        inside=x.norm(dim=-1)<cfg.bound_radius
        distance=torch.cdist(f*inside,f*inside,p=1)
        distance.fill_diagonal_(float('inf'))
        # A few spare edges cover numerical ties between the prepass and the live microbatch.
        target=distance.topk(min(n-1,cfg.nearest_k+4),largest=False).indices
        self.dst=target.flatten()
        self.src=torch.arange(n,device=x.device).repeat_interleave(target.shape[1])
        coord_a,coord_b,rotations,masks=[],[],[],[]
        for aa,bb in zip(self.src.split(48),self.dst.split(48)):
            fa,fb=f[aa],f[bb]
            R,t,scale=_frame(x,fa,fb,g[bb],inside)
            band=(torch.minimum(fa.abs(),fb.abs())<.04)&inside
            # The common probe order is already random, independent of shape.
            ix=(torch.arange(m,device=x.device)+m*(~band).long()).topk(min(16,m),largest=False).indices
            p=x[ix]
            coord_a.append(p)
            coord_b.append((p @ R.transpose(1,2))/scale[:,None,None]+t[:,None])
            rotations.append(R)
            masks.append(band.gather(1,ix))
        self.rotation=torch.cat(rotations)
        self.pair_points=torch.cat([torch.cat(coord_a),torch.cat(coord_b)])
        ids=torch.cat([self.src,self.dst])
        # Pack by code instead of expanding the decoder weights separately for every edge.
        order=ids.argsort(stable=True)
        counts=ids.bincount(minlength=n)
        within=torch.arange(len(ids),device=x.device)-torch.repeat_interleave(counts.cumsum(0)-counts,counts)
        self.slots=torch.empty_like(order)
        self.slots[order]=within
        self.ids=ids
        self.packed=x.new_zeros(n,int(counts.max()),self.pair_points.shape[1],3)
        self.packed[ids,self.slots]=self.pair_points
        field=x.new_empty(self.packed.shape[:-1])
        slopes=torch.empty_like(self.packed)
        for cs,ps,mod,coordinates in self.blocks():
            values,grad=field_and_grad(model,mod,coordinates,create_graph=False)
            field.flatten(1)[cs,ps]=values.detach()
            slopes.flatten(1,2)[cs,ps]=grad.detach()
        self.gradients=slopes[ids,self.slots]
        fa,fb=field[ids,self.slots].chunk(2)
        na,nb=self.gradients.norm(dim=-1).chunk(2)
        self.valid=torch.cat(masks)&(torch.minimum(fa.abs(),fb.abs())<.04)&(na>.5)&(nb>.5)
        angles=self.angles(self.gradients)
        self.table=x.new_zeros(n,n)
        self.table[self.src,self.dst]=angles
        self.table.requires_grad_()
        self.mean_angle=float(angles.mean())
        self.mean_sites=float(self.valid.sum(1).float().mean())

    def blocks(self):
        flat=self.packed.flatten(1,2)
        for i in range(0,len(self.codes),4):
            cs=slice(i,i+4)
            for j in range(0,flat.shape[1],256):
                ps=slice(j,j+256)
                yield cs,ps,self.model.prepare(self.codes[cs]),flat[cs,ps]

    def angles(self, gradients):
        a,b=F.normalize(gradients,dim=-1,eps=1.e-6).chunk(2)
        b=b @ self.rotation  # gradient of B at the aligned location, in A's frame
        angle=.5*(a-b).square().sum(-1)
        return (angle*self.valid).sum(-1)/self.valid.sum(-1).clamp_min(1)

    def backward(self):
        # Each owner row supplies its upstream derivative exactly once across all passes.
        grad=self.gradients.detach().requires_grad_()
        weights=self.table.grad[self.src,self.dst]
        (vjp,)=torch.autograd.grad(self.angles(grad),grad,grad_outputs=weights)
        packed=torch.zeros_like(self.packed)
        packed[self.ids,self.slots]=vjp.detach()
        packed=packed.flatten(1,2)
        for cs,ps,mod,coordinates in self.blocks():
            _,g=field_and_grad(self.model,mod,coordinates,create_graph=True)
            torch.autograd.backward(g,grad_tensors=packed[cs,ps])
