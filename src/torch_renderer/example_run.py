import argparse
import torch
import imageio.v3 as iio

from soft_renderer import (
    Plane,
    create_camera_from_mitsuba,
    DiffuseBRDF,
    render,
    Texture,
)

# this file is not maintained and is not used in the project. So it might not work as we develop the renderer.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texture", default=None, help="Deprecated; use --brdf_texture. Ignored by geometry.")
    parser.add_argument("--plane_w", type=float, default=2.0)
    parser.add_argument("--plane_h", type=float, default=2.0)
    parser.add_argument("--out", default="render_soft.png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--eye", type=float, nargs=3, default=[0.0, 0.0, 3.0])
    parser.add_argument("--at", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    parser.add_argument("--light", type=float, nargs=3, default=[0.0, 0.5, 1.5])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no_filter", action="store_true", help="Disable Gaussian reconstruction filter")
    parser.add_argument("--filter_stddev", type=float, default=0.5)
    parser.add_argument("--filter_radius", type=float, default=2.0)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--no_jitter", action="store_true", help="Disable per-pixel jitter")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--brdf_texture", type=str, default=None, help="Path to diffuse texture (linearized from sRGB)")
    parser.add_argument("--albedo", type=float, nargs=3, default=[0.8, 0.8, 0.8], help="Constant diffuse reflectance if no texture")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Geometry
    plane = Plane.create(width=args.plane_w, height=args.plane_h, device=device)

    # Camera (Mitsuba-like sensor)
    # Build Mitsuba-style to_world from eye/at/up if user didn't provide one
    eye = torch.tensor(args.eye, dtype=torch.float32, device=device)
    at = torch.tensor(args.at, dtype=torch.float32, device=device)
    up = torch.tensor(args.up, dtype=torch.float32, device=device)
    forward = torch.nn.functional.normalize(at - eye, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    up_vec = torch.nn.functional.normalize(torch.cross(right, forward, dim=0), dim=0)
    # Mitsuba camera-to-world: columns are (-right, up, forward), last column is translation
    to_world = torch.eye(4, dtype=torch.float32, device=device)
    to_world[:3, 0] = -right
    to_world[:3, 1] = up_vec
    to_world[:3, 2] = forward
    to_world[:3, 3] = eye

    sensor = {
        "fov": args.fov,
        "resolution": (args.height, args.width),
        "to_world": to_world.cpu().tolist(),
    }
    camera = create_camera_from_mitsuba(sensor, device=device)

    # BRDF (textured or constant)
    # Backwards-compat: if --texture provided and --brdf_texture not set, use it for BRDF
    if args.brdf_texture is None and args.texture is not None:
        args.brdf_texture = args.texture

    if args.brdf_texture is not None:
        brdf = DiffuseBRDF(texture_path=args.brdf_texture, device=device, srgb=True)
    else:
        brdf = DiffuseBRDF(reflectance=tuple(args.albedo), device=device)

    # Render
    outputs = render(
        camera=camera,
        plane=plane,
        brdf=brdf,
        image_size=(args.height, args.width),
        light_position=torch.tensor(args.light, dtype=torch.float32, device=device),
        light_intensity=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device),
        apply_filter=not args.no_filter,
        filter_stddev=args.filter_stddev,
        filter_radius=args.filter_radius,
        spp=args.spp,
        jitter=not args.no_jitter,
        seed=args.seed,
    )

    image = outputs["color"] if isinstance(outputs, dict) else outputs
    iio.imwrite(args.out, (image.cpu().numpy() * 255).astype("uint8"))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()


