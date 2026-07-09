from src.neumip_nets import load_network
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr

import torch
from src.utils import vec_to_tens_safe



def load_latents(path, channels, latents_vector = None):
    if path is None:
        return [mi.load_dict({'type': 'bitmap', 'raw': True, 'filter_type': 'bilinear' , 'data': mi.TensorXf(latents_vector[...,i:i+1])}) for i in range(channels)]
    return [mi.load_dict({'type': 'bitmap', 'raw': True, 'filter_type': 'bilinear' , 'filename': f'{path}/latent{i}.exr'}) for i in range(channels)]



class MitsubaTorchWrapperMLP():
    # A wrapper to wrap torch model in mitsuba tensors (with tricks to enable gradients for the torch model)
    def __init__(
        self,
        desc_path
    ):
        self.grad_activator = mi.Vector3f(0)

        self.network = load_network(desc_path, 'cuda')

    def eval(self, wi, wo, uvs):

        # NOTE: Important: in mitsuba wi is the camera ray direction, and wo is the sampled direction
        wi, wo = wo, wi


        wi_tensor = vec_to_tens_safe(wi )
        wo_tensor = vec_to_tens_safe(wo + self.grad_activator)
        uvs_tensor = vec_to_tens_safe(uvs)
        net_out = self.eval_torch(wi_tensor, wo_tensor, uvs_tensor)

        output = dr.unravel(mi.Vector3f, net_out.array)
        return output

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, wi, wo, uvs):
        return self.network(wi, wo, uvs)







class NeuralBSDF(mi.BSDF):
    # the actual NeuMIP BSDF
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.latents: list[mi.TensorXf] = None
        self.m_flags =  mi.BSDFFlags.FrontSide | mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.Anisotropic | mi.BSDFFlags.SpatiallyVarying
        self.network = None

        self.m_components = [self.m_flags]

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

    def traverse(self, callback):
        if (self.latents is not None):
            for i, l in enumerate(self.latents):
                data = mi.traverse(l)['data']                
                callback.put_parameter(f'latent_{i}', data, mi.ParamFlags.Differentiable)
        callback.put_parameter("network", self.network.grad_activator, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        if (self.latents is not None):
            for l in self.latents:
                l.parameters_changed()

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


