from .legacy_neumip import load_network
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr

import torch
from src.utils import vec_to_tens_safe

import numpy as np

# def load_latents(path, channels, uv_scale = 1, is_offset = False):
#     return [mi.load_dict({'type': 'bitmap', 'raw': True, "to_uv": mi.ScalarTransform3f.scale(uv_scale),'filter_type': 'bilinear' , 'filename': f'{path}/{'offset_' if is_offset else ''}latent_{i*3}-{(i+1)*3}.exr'}) for i in range(channels // 3)]

def load_latents(path, channels, uv_scale, uv_shift, is_offset = False, numpy = False):
    bitmaps = [mi.TensorXf(mi.Bitmap(f'{path}/{'offset_' if is_offset else ''}latent_{i*3}-{(i+1)*3}.exr')).numpy() for i in range(channels // 3)]
    if numpy:
        return bitmaps
    bitmaps = [np.flip(b, axis=0) for b in bitmaps]
    return [mi.load_dict({'type': 'bitmap', 'raw': True, "format": 'variant','wrap_mode': 'mirror' , "to_uv": mi.ScalarTransform4f.scale(uv_scale).translate([uv_shift[0],uv_shift[1],0]).rotate([0,0,1], 20),'filter_type': 'bilinear', 'data': mi.TensorXf(bitmaps[i])}) for i in range(channels // 3)]



class MitsubaTorchWrapperMLP():
    # A wrapper to wrap torch model in mitsuba tensors (with tricks to enable gradients for the torch model)
    def __init__(
        self,
        desc_path
    ):
        self.grad_activator = mi.Vector3f(0)
        self.batch_size = 2**22
        self.network = load_network(desc_path, 'cuda')

    def eval(self, wi, wo, uvs):

        wi_tensor = vec_to_tens_safe(wi )
        wo_tensor = vec_to_tens_safe(wo + self.grad_activator)
        uvs_tensor = vec_to_tens_safe(uvs)
        net_out = self.eval_torch(wi_tensor, wo_tensor, uvs_tensor)

        output = dr.unravel(mi.Vector3f, net_out.array)
        return output

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, wi, wo, uvs):
        B = wi.shape[0]
        if B <= self.batch_size:
            return self.network(wi, wo, uvs)
        print(f'Evaluating in batches of {self.batch_size}, current batch size: {B}')
        # Process in batches
        outputs = []
        for start in range(0, B, self.batch_size):
            end = min(start + self.batch_size, B)
            wi_batch = wi[start:end]
            wo_batch = wo[start:end]
            uvs_batch = uvs[start:end]
            out_batch = self.network(wi_batch, wo_batch, uvs_batch)
            outputs.append(out_batch)
        
        return torch.cat(outputs, dim=0)







class NeuralBSDF(mi.BSDF):
    # the actual NeuMIP BSDF
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.latent_textures = []
        self.offset_latent_textures = []
        self.m_flags =  mi.BSDFFlags.FrontSide | mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.Anisotropic | mi.BSDFFlags.SpatiallyVarying
        self.network = None
        self.m_components = [self.m_flags]
        self.use_coop = True

    def sample(self, ctx, si, sample1, sample2, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = +mi.BSDFFlags.GlossyReflection
        bs.sampled_component = 0

        value = self.eval(ctx, si, bs.wo, active) / bs.pdf

        return (bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        wi, wo = self.transform(si.wi, wo)        
        wo, wi = wi, wo # NOTE: Important: in mitsuba wi is the camera ray direction, and wo is the sampled direction
        if self.use_coop:
            value = self.network.eval(wi, wo, si.uv, self.latent_textures, self.offset_latent_textures, si, active)
        else:
            value = self.network.eval(wi, wo, si.uv)

        # Assumption: the cos theta is not already accounted for in the MLP output
        value = value * cos_theta_o
        # Assumption: no need to divide by 1/PI
        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, mi.Vector3f(0))



    def eval_diffuse_reflectance(self, si, active = True):
        return si.p * 0




    def eval_pdf(self, ctx, si, wo, active):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def pdf(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return mi.Vector3f(0)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)



    def transform(self, wi, wo):
        wi = dr.normalize(wi)
        wo = dr.normalize(wo)
        return wi, wo

    def to_string(self):
        return ('neural[\n'
                ']')
    



class NeuralBSDFOffsetOnly(NeuralBSDF):
    # This is used to render an image with the offset only
    # the current design is dirty, requiring setting return_offset to True, and having a special integrator (OffsetIntegrator)
    def eval(self, ctx, si, wo, active):
        cos_theta_o = mi.Frame3f.cos_theta(wo) # to undo the cosine term
        return dr.select((cos_theta_o > 0.0), super().eval(ctx, si, wo, active) / cos_theta_o, mi.Vector3f(0))


