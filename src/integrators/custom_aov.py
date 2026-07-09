import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from typing import Tuple, List, Any, Optional
import drjit as dr
import gc

class CenterPixelSamplerIntegrator(mi.python.ad.integrators.common.ADIntegrator):
    # this integrator inherits from the mitsuba built-in ADIntegrator with one difference, 
    # it disables the random offset for antialising and only samples from the center of the pixel
    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray
        """
        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()
        assert spp == 1 #NOTE: We assume 1 spp for now

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(mi.UInt32(mi.Int32(-film_size[0])), pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += mi.Vector2i(film.crop_offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + 0.5                      #NOTE: other than 1spp, the only difference to mitsuba built-in code is here, instead of random offset, we sample the center of the pixel

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

        return ray, weight, splatting_pos

class ArbitraryIntegrator(CenterPixelSamplerIntegrator):

    def __init__(self, props = mi.Properties()):
        super().__init__(props)
        self.arb_vals = {}
        

    def clear_arb_vals(self):
        del self.arb_vals
        self.arb_vals = {}
        dr.flush_malloc_cache()
        dr.flush_malloc_cache()
        gc.collect()


    def get_arb_vals(self, w, h):
        vals = {k: v.numpy().transpose(1,0) for k, v in self.arb_vals.items()}
        vals = {k: v.reshape(w,h, v.shape[-1]) for k, v in vals.items()}        
        self.clear_arb_vals()
        return vals

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               δaovs: Optional[mi.Spectrum],
               state_in: Any,
               active: mi.Bool) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float]]:


        si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
        result = mi.Spectrum(0.0)

        # NEE
        ctx = mi.BSDFContext()
        bsdf = si.bsdf(ray)
        flags = bsdf.flags()
        active = active & si.is_valid()

        sample_emitter = active & mi.has_flag(flags, mi.BSDFFlags.Smooth)

        active_e = sample_emitter
        (ds, emitter_val) = scene.sample_emitter_direction(si, sampler.next_2d(active_e), True, active_e)
        active_e = active & (ds.pdf != 0.0)

        # Query the BSDF for that emitter-sampled direction
        wo = si.to_local(ds.d)

        bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active_e)
        # bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)
        # mis = dr.select(ds.delta, mi.Float(1.0), self.mis_weight(
        #     ds.pdf * 1, bsdf_pdf * 0) * 1);
        mis = 1.0

        result[active_e] += mis * bsdf_val * emitter_val


        self.arb_vals['w_camera'] = dr.select(active, si.to_local(-ray.d), mi.Vector3f(0))
        self.arb_vals['uv'] = dr.select(active, si.uv, mi.Vector2f(0))
        self.arb_vals['valid_light_sample'] = dr.select(active_e, mi.Vector1f(1), mi.Vector1f(0))
        self.arb_vals['bsdf_val'] = dr.select(active_e, bsdf_val, mi.Vector3f(0))
        self.arb_vals['w_light'] = dr.select(active_e, wo, mi.Vector3f(0))
        self.arb_vals['bsdf_pdf'] = dr.select(active_e, bsdf_pdf, mi.Vector3f(0))
        self.arb_vals['radiance'] = dr.select(active_e, result, mi.Vector3f(0))
        self.arb_vals['emitter_val'] = dr.select(active_e, emitter_val, mi.Vector3f(0))



        return result, active, [], None
