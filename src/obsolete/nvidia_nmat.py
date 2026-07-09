from src.nvidia_neural_material import load_network
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr

import torch



def load_latents(path, channels):
    return [mi.load_dict({'type': 'bitmap', 'raw': True, 'filter_type': 'nearest' , 'filename': f'{path}/latent{i}.exr'}) for i in range(channels)]



class MitsubaTorchWrapper():
    def __init__(
        self,
        desc_path
    ):
        self.network = load_network(desc_path, 'cuda')
        self.network = self.network

    def eval(self, wi, wo, latents):
        wi_tensor = vec_to_tens_safe(wi)
        wo_tensor = vec_to_tens_safe(wo)
        if isinstance(self.network, torch.nn.Module):
            net_out = self.eval_torch(wi_tensor, wo_tensor, latents)
        else:
            net_out = self.network(wi_tensor, wo_tensor, latents)

        output = dr.unravel(mi.Vector3f, net_out.array)
        return output

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, wi, wo, latents):
        return self.network(wi, wo, latents)


class NeuralBSDF(mi.BSDF):
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

        latents = [l.eval_1(si, active) for l in self.latents]
        
        latents = concat_1d_latents(latents)
        wh, wd = self.transform(si.wi, wo)
        # I did not replace the wo with wd in place ... So the caller of this function will trace based on the original wo

        value = self.network.eval(wh, wd, latents)
        # Assumption: the cos theta is already accounted for in the MLP output
        # value = value * cos_theta_o
        # Assumption: no need to divide by 1/PI

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, mi.Vector3f(0))

    def transform(self, wi, wo):
        wh = dr.normalize(wo + wi)
        whSph = cartesian_to_spherical_rad(wh)
        temp = rotate_z(wi, -whSph.y)
        wd = rotate_y(temp, -whSph.x)
        return wh,wd


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

    def parameters_changed(self, keys):
        if (self.latents is not None):
            for l in self.latents:
                l.parameters_changed()

    def to_string(self):
        return ('neural[\n'
                ']')
    
# mi.register_bsdf("neural", lambda props: NeuralBSDF(props))


# ______Helper functions ______

#/**
# * Rotate a 3D vector counterclockwise around the X-axis.
# * The rotation angle is positive for a rotation that is counterclockwise when looking along the x-axis towards the origin.
# * @param[in] v Original vector.
# * @param[in] angle Angle in radians.
# * @return Transformed vector.
# */
def rotate_x(v, angle):
    s = dr.sin(angle)
    c = dr.cos(angle)
    return type(v)(v.x, v.y * c - v.z * s, v.y * s + v.z * c)

# /**
#  * Rotate a 3D vector counterclockwise around the Y-axis.
#  * The rotation angle is positive for a rotation that is counterclockwise when looking along the y-axis towards the origin.
#  * @param[in] v Original vector.
#  * @param[in] angle Angle in radians.
#  * @return Transformed vector.
#  */
def rotate_y(v, angle):
    s = dr.sin(angle)
    c = dr.cos(angle)
    return type(v)(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)

# /**
#  * Rotate a 3D vector counterclockwise around the Z-axis.
#  * The rotation angle is positive for a rotation that is counterclockwise when looking along the z-axis towards the origin.
#  * @param[in] v Original vector.
#  * @param[in] angle Angle in radians.
#  * @return Transformed vector.
#  */
def rotate_z(v,  angle):
    s = dr.sin(angle)
    c = dr.cos(angle)
    return type(v)(v.x * c - v.y * s, v.x * s + v.y * c, v.z)




def cartesian_to_spherical_rad(p):
    p = dr.normalize(p)
    shpe = mi.Vector2f()
    shpe.x = dr.acos(p.z)
    shpe.y = dr.atan2(-p.y, -p.x) + dr.pi
    return shpe




def vec_to_tens_safe(vec):
    # Converts a Vector3f to a TensorXf safely in mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(dr.ravel(vec), shape=(dr.shape(vec)[1], dr.shape(vec)[0]))


def concat_1d_latents(latents):
    channels = len(latents)
    n_rows = len(latents[0])
    all_latents = dr.empty(mi.Float, n_rows*channels)
    inds = dr.arange(mi.UInt32, n_rows) * channels
    for i in range(channels):
        dr.scatter(all_latents, latents[i], inds+i)
    return mi.TensorXf(all_latents, shape=(n_rows,channels))