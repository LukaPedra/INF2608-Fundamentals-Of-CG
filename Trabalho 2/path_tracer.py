import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image


EPSILON = 1e-4
PI = math.pi


def vec3(value):
    return np.array(value, dtype=float)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0.0 else v


def dot(a, b):
    return float(np.dot(a, b))


def cross(a, b):
    return np.cross(a, b)


def luminance(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def make_basis(n):
    n = normalize(n)
    if abs(n[0]) > 0.9:
        tangent = normalize(cross([0.0, 1.0, 0.0], n))
    else:
        tangent = normalize(cross([1.0, 0.0, 0.0], n))
    bitangent = cross(n, tangent)
    return tangent, bitangent, n


def cosine_sample_hemisphere(normal, rng):
    xi1 = rng.random()
    xi2 = rng.random()
    r = math.sqrt(xi1)
    phi = 2.0 * PI * xi2
    local = np.array([
        r * math.cos(phi),
        r * math.sin(phi),
        math.sqrt(max(0.0, 1.0 - xi1)),
    ])
    tangent, bitangent, n = make_basis(normal)
    direction = local[0] * tangent + local[1] * bitangent + local[2] * n
    pdf = max(0.0, dot(normalize(direction), normal)) / PI
    return normalize(direction), pdf


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def __init__(self, origin, direction):
        self.origin = vec3(origin)
        self.direction = normalize(vec3(direction))


@dataclass
class Material:
    albedo: np.ndarray
    emission: np.ndarray

    def __init__(self, albedo=(1.0, 1.0, 1.0), emission=(0.0, 0.0, 0.0)):
        self.albedo = vec3(albedo)
        self.emission = vec3(emission)

    @property
    def is_emissive(self):
        return luminance(self.emission) > 0.0

    def brdf(self):
        return self.albedo / PI


@dataclass
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    material: Material
    obj: object


class Sphere:
    def __init__(self, center, radius, material):
        self.center = vec3(center)
        self.radius = float(radius)
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = dot(ray.direction, ray.direction)
        b = 2.0 * dot(oc, ray.direction)
        c = dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sq = math.sqrt(disc)
        t = (-b - sq) / (2.0 * a)
        if t <= EPSILON:
            t = (-b + sq) / (2.0 * a)
        if t <= EPSILON:
            return None
        p = ray.origin + t * ray.direction
        n = normalize(p - self.center)
        return Hit(t, p, n, self.material, self)


class Box:
    def __init__(self, bmin, bmax, material):
        self.bmin = vec3(bmin)
        self.bmax = vec3(bmax)
        self.material = material

    def intersect(self, ray):
        inv_dir = np.empty(3)
        for i in range(3):
            inv_dir[i] = 1.0 / ray.direction[i] if abs(ray.direction[i]) > 1e-12 else 1e30
        t0 = (self.bmin - ray.origin) * inv_dir
        t1 = (self.bmax - ray.origin) * inv_dir
        t_near = np.minimum(t0, t1)
        t_far = np.maximum(t0, t1)
        tmin = float(np.max(t_near))
        tmax = float(np.min(t_far))
        if tmin > tmax or tmax <= EPSILON:
            return None
        t = tmin if tmin > EPSILON else tmax
        p = ray.origin + t * ray.direction
        return Hit(t, p, self._normal_at(p), self.material, self)

    def _normal_at(self, p):
        center = 0.5 * (self.bmin + self.bmax)
        half = 0.5 * (self.bmax - self.bmin)
        gaps = half - np.abs(p - center)
        axis = int(np.argmin(gaps))
        n = np.zeros(3)
        n[axis] = 1.0 if p[axis] >= center[axis] else -1.0
        return n


class Plane:
    def __init__(self, point, normal, material):
        self.point = vec3(point)
        self.normal = normalize(vec3(normal))
        self.material = material

    def intersect(self, ray):
        denom = dot(self.normal, ray.direction)
        if abs(denom) < 1e-8:
            return None
        t = dot(self.point - ray.origin, self.normal) / denom
        if t <= EPSILON:
            return None
        return Hit(t, ray.origin + t * ray.direction, self.normal, self.material, self)


class Triangle:
    def __init__(self, a, b, c, material):
        self.a = vec3(a)
        self.b = vec3(b)
        self.c = vec3(c)
        self.material = material
        self.normal = normalize(cross(self.b - self.a, self.c - self.a))
        self.area = 0.5 * np.linalg.norm(cross(self.b - self.a, self.c - self.a))

    def intersect(self, ray):
        edge1 = self.b - self.a
        edge2 = self.c - self.a
        h = cross(ray.direction, edge2)
        det = dot(edge1, h)
        if abs(det) < 1e-8:
            return None
        inv_det = 1.0 / det
        s = ray.origin - self.a
        u = inv_det * dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = cross(s, edge1)
        v = inv_det * dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = inv_det * dot(edge2, q)
        if t <= EPSILON:
            return None
        n = self.normal if dot(self.normal, ray.direction) < 0.0 else -self.normal
        return Hit(t, ray.origin + t * ray.direction, n, self.material, self)


class RectAreaLight:
    def __init__(self, center, edge_u, edge_v, radiance):
        self.center = vec3(center)
        self.edge_u = vec3(edge_u)
        self.edge_v = vec3(edge_v)
        self.radiance = vec3(radiance)
        self.material = Material(emission=self.radiance)
        self.normal = normalize(cross(self.edge_u, self.edge_v))
        self.area = np.linalg.norm(cross(self.edge_u, self.edge_v))

    def sample(self, rng):
        u = rng.random() - 0.5
        v = rng.random() - 0.5
        point = self.center + u * self.edge_u + v * self.edge_v
        return point, self.normal, self.radiance, 1.0 / self.area

    def intersect(self, ray):
        denom = dot(self.normal, ray.direction)
        if abs(denom) < 1e-8:
            return None
        t = dot(self.center - ray.origin, self.normal) / denom
        if t <= EPSILON:
            return None
        p = ray.origin + t * ray.direction
        rel = p - self.center
        u_len2 = dot(self.edge_u, self.edge_u)
        v_len2 = dot(self.edge_v, self.edge_v)
        u = dot(rel, self.edge_u) / u_len2
        v = dot(rel, self.edge_v) / v_len2
        if abs(u) > 0.5 or abs(v) > 0.5:
            return None
        n = self.normal if dot(self.normal, ray.direction) < 0.0 else -self.normal
        return Hit(t, p, n, self.material, self)

    def pdf_solid_angle(self, point, direction):
        hit = self.intersect(Ray(point + direction * EPSILON, direction))
        if hit is None:
            return 0.0
        cos_light = max(0.0, dot(self.normal, -direction))
        if cos_light <= 0.0:
            return 0.0
        dist2 = dot(hit.point - point, hit.point - point)
        return dist2 / (cos_light * self.area)


class MeshAreaLight:
    def __init__(self, triangles, radiance):
        self.radiance = vec3(radiance)
        self.material = Material(emission=self.radiance)
        self.triangles = [Triangle(a, b, c, self.material) for a, b, c in triangles]
        self.areas = np.array([tri.area for tri in self.triangles])
        self.total_area = float(np.sum(self.areas))
        self.cdf = np.cumsum(self.areas) / self.total_area

    def sample(self, rng):
        idx = int(np.searchsorted(self.cdf, rng.random(), side="right"))
        tri = self.triangles[min(idx, len(self.triangles) - 1)]
        r1 = math.sqrt(rng.random())
        r2 = rng.random()
        point = (1.0 - r1) * tri.a + r1 * (1.0 - r2) * tri.b + r1 * r2 * tri.c
        return point, tri.normal, self.radiance, 1.0 / self.total_area

    def intersect(self, ray):
        closest = None
        for tri in self.triangles:
            hit = tri.intersect(ray)
            if hit and (closest is None or hit.t < closest.t):
                closest = hit
        return closest

    def pdf_solid_angle(self, point, direction):
        hit = self.intersect(Ray(point + direction * EPSILON, direction))
        if hit is None:
            return 0.0
        light_normal = hit.obj.normal
        cos_light = max(0.0, dot(light_normal, -direction))
        if cos_light <= 0.0:
            return 0.0
        dist2 = dot(hit.point - point, hit.point - point)
        return dist2 / (cos_light * self.total_area)


class EnvironmentLight:
    def __init__(self, top=(0.45, 0.58, 0.85), bottom=(0.8, 0.82, 0.9), strength=0.7):
        self.top = vec3(top) * strength
        self.bottom = vec3(bottom) * strength

    def radiance(self, direction):
        t = 0.5 * (normalize(direction)[1] + 1.0)
        return (1.0 - t) * self.bottom + t * self.top


class Scene:
    def __init__(self, environment=None):
        self.objects = []
        self.lights = []
        self.environment = environment

    def add(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)
        self.objects.append(light)

    def closest_intersection(self, ray):
        closest = None
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and (closest is None or hit.t < closest.t):
                closest = hit
        return closest

    def visible(self, origin, target):
        direction = target - origin
        dist = np.linalg.norm(direction)
        if dist <= EPSILON:
            return False
        direction = direction / dist
        hit = self.closest_intersection(Ray(origin + direction * EPSILON, direction))
        return hit is None or hit.t >= dist - 2.0 * EPSILON or hit.material.is_emissive

    def sample_light(self, rng):
        if not self.lights:
            return None
        idx = rng.integers(0, len(self.lights))
        light = self.lights[int(idx)]
        point, normal, radiance, pdf_area = light.sample(rng)
        return light, point, normal, radiance, pdf_area / len(self.lights)

    def light_pdf_solid_angle(self, point, direction):
        if not self.lights:
            return 0.0
        pdf = 0.0
        for light in self.lights:
            pdf += light.pdf_solid_angle(point, direction)
        return pdf / len(self.lights)


def direct_light_sample(scene, hit, rng, use_mis=False):
    sample = scene.sample_light(rng)
    if sample is None:
        return np.zeros(3)
    _, light_point, light_normal, light_radiance, pdf_area = sample
    to_light = light_point - hit.point
    dist2 = dot(to_light, to_light)
    dist = math.sqrt(dist2)
    wi = to_light / dist
    cos_surface = max(0.0, dot(hit.normal, wi))
    cos_light = max(0.0, dot(light_normal, -wi))
    if cos_surface <= 0.0 or cos_light <= 0.0:
        return np.zeros(3)
    if not scene.visible(hit.point + hit.normal * EPSILON, light_point):
        return np.zeros(3)
    pdf_light = pdf_area * dist2 / cos_light
    if pdf_light <= 0.0:
        return np.zeros(3)
    weight = 1.0
    if use_mis:
        pdf_brdf = cos_surface / PI
        weight = pdf_light / (pdf_light + pdf_brdf) if pdf_light + pdf_brdf > 0.0 else 0.0
    return weight * hit.material.brdf() * light_radiance * cos_surface / pdf_light


def brdf_direct_sample(scene, hit, rng):
    wi, pdf_brdf = cosine_sample_hemisphere(hit.normal, rng)
    if pdf_brdf <= 0.0:
        return np.zeros(3)
    shadow_hit = scene.closest_intersection(Ray(hit.point + hit.normal * EPSILON, wi))
    incoming = np.zeros(3)
    if shadow_hit is None:
        if scene.environment is not None:
            incoming = scene.environment.radiance(wi)
        else:
            return np.zeros(3)
    elif shadow_hit.material.is_emissive:
        incoming = shadow_hit.material.emission
    else:
        return np.zeros(3)
    pdf_light = scene.light_pdf_solid_angle(hit.point, wi)
    weight = pdf_brdf / (pdf_brdf + pdf_light) if pdf_brdf + pdf_light > 0.0 else 1.0
    cos_surface = max(0.0, dot(hit.normal, wi))
    return weight * hit.material.brdf() * incoming * cos_surface / pdf_brdf


def trace_path(ray, scene, rng, max_depth=4, use_mis=False):
    radiance = np.zeros(3)
    throughput = np.ones(3)

    for depth in range(max_depth):
        hit = scene.closest_intersection(ray)
        if hit is None:
            if scene.environment is not None:
                radiance += throughput * scene.environment.radiance(ray.direction)
            break

        if hit.material.is_emissive:
            if depth == 0 or not use_mis:
                radiance += throughput * hit.material.emission
            break

        direct = direct_light_sample(scene, hit, rng, use_mis=use_mis)
        if use_mis:
            direct += brdf_direct_sample(scene, hit, rng)
        radiance += throughput * direct

        wi, pdf = cosine_sample_hemisphere(hit.normal, rng)
        if pdf <= 0.0:
            break
        cos_theta = max(0.0, dot(hit.normal, wi))
        throughput *= hit.material.brdf() * cos_theta / pdf
        ray = Ray(hit.point + hit.normal * EPSILON, wi)

    return radiance


class Camera:
    def __init__(self, position, target, up=(0.0, 1.0, 0.0), fov=45.0, aspect=1.0):
        self.position = vec3(position)
        self.forward = normalize(vec3(target) - self.position)
        self.right = normalize(cross(self.forward, vec3(up)))
        self.up = normalize(cross(self.right, self.forward))
        self.scale = math.tan(math.radians(fov) * 0.5)
        self.aspect = aspect

    def generate_ray(self, u, v):
        x = (2.0 * u - 1.0) * self.aspect * self.scale
        y = (1.0 - 2.0 * v) * self.scale
        direction = normalize(self.forward + x * self.right + y * self.up)
        return Ray(self.position, direction)


def render(scene_builder, filename, width, height, spp, max_depth, seed, use_mis):
    rng = np.random.default_rng(seed)
    scene, camera = scene_builder(width / height)
    image = np.zeros((height, width, 3), dtype=float)
    start = time.time()
    print(f"Rendering {filename}: {width}x{height}, spp={spp}, depth={max_depth}, mis={use_mis}")

    for y in range(height):
        if y % max(1, height // 10) == 0:
            print(f"  line {y}/{height}")
        for x in range(width):
            color = np.zeros(3)
            for _ in range(spp):
                u = (x + rng.random()) / width
                v = (y + rng.random()) / height
                color += trace_path(camera.generate_ray(u, v), scene, rng, max_depth, use_mis)
            image[y, x] = color / spp

    mapped = np.power(np.clip(image, 0.0, None) / (1.0 + np.clip(image, 0.0, None)), 1.0 / 2.2)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    Image.fromarray(np.clip(mapped * 255.0, 0, 255).astype(np.uint8)).save(filename)
    print(f"Saved {filename} in {time.time() - start:.2f}s")


def cornell_materials():
    return {
        "white": Material((0.78, 0.78, 0.72)),
        "red": Material((0.75, 0.15, 0.10)),
        "green": Material((0.12, 0.55, 0.18)),
        "blue": Material((0.15, 0.25, 0.75)),
        "yellow": Material((0.75, 0.65, 0.18)),
    }


def add_cornell_box(scene, open_top=False, open_back=False):
    mat = cornell_materials()
    scene.add(Plane((0, 0, 0), (0, 1, 0), mat["white"]))
    if not open_top:
        scene.add(Plane((0, 6, 0), (0, -1, 0), mat["white"]))
    if not open_back:
        scene.add(Plane((0, 0, -4), (0, 0, 1), mat["white"]))
    scene.add(Plane((-3, 0, 0), (1, 0, 0), mat["red"]))
    scene.add(Plane((3, 0, 0), (-1, 0, 0), mat["green"]))
    return mat


def base_scene(aspect):
    scene = Scene()
    mat = add_cornell_box(scene)
    scene.add(Sphere((-1.05, 1.0, -1.4), 1.0, mat["blue"]))
    scene.add(Box((0.55, 0.0, -2.7), (1.8, 2.2, -1.45), mat["yellow"]))
    scene.add_light(RectAreaLight((0.0, 5.85, -1.3), (1.7, 0.0, 0.0), (0.0, 0.0, 1.7), (12.0, 12.0, 11.0)))
    camera = Camera((0, 3.0, 7.8), (0, 2.7, -1.5), aspect=aspect, fov=42.0)
    return scene, camera


def environment_scene(aspect):
    scene = Scene(EnvironmentLight(strength=0.9))
    mat = add_cornell_box(scene, open_top=True, open_back=True)
    scene.add(Sphere((-1.1, 1.0, -1.2), 1.0, mat["blue"]))
    scene.add(Box((0.55, 0.0, -2.6), (1.7, 1.9, -1.45), mat["white"]))
    scene.add_light(RectAreaLight((0.0, 5.4, -1.0), (1.5, 0.0, 0.0), (0.0, 0.0, 1.5), (3.5, 3.5, 3.3)))
    camera = Camera((0, 3.0, 7.8), (0, 2.6, -1.5), aspect=aspect, fov=42.0)
    return scene, camera


def tetrahedron(center, scale):
    c = vec3(center)
    verts = [
        c + scale * vec3((0.0, 0.9, 0.0)),
        c + scale * vec3((-0.8, -0.45, 0.65)),
        c + scale * vec3((0.8, -0.45, 0.65)),
        c + scale * vec3((0.0, -0.45, -0.85)),
    ]
    return [
        (verts[0], verts[2], verts[1]),
        (verts[0], verts[1], verts[3]),
        (verts[0], verts[3], verts[2]),
        (verts[1], verts[2], verts[3]),
    ]


def mesh_light_scene(aspect):
    scene = Scene()
    mat = add_cornell_box(scene)
    scene.add(Sphere((-1.05, 1.0, -1.4), 1.0, mat["white"]))
    scene.add(Box((0.65, 0.0, -2.8), (1.8, 1.7, -1.5), mat["blue"]))
    scene.add_light(MeshAreaLight(tetrahedron((0.0, 5.05, -1.2), 0.7), (10.0, 8.5, 6.0)))
    camera = Camera((0, 3.0, 7.8), (0, 2.6, -1.5), aspect=aspect, fov=42.0)
    return scene, camera


def mis_scene(aspect):
    scene, camera = base_scene(aspect)
    scene.lights.clear()
    scene.objects = [obj for obj in scene.objects if not isinstance(obj, RectAreaLight)]
    scene.add_light(RectAreaLight((0.0, 5.85, -1.3), (0.65, 0.0, 0.0), (0.0, 0.0, 0.65), (38.0, 38.0, 36.0)))
    return scene, camera


SCENES = {
    "base_cornell": base_scene,
    "environment_light": environment_scene,
    "mesh_light": mesh_light_scene,
    "mis_compare": mis_scene,
}


def render_suite(args):
    out_dir = args.output_dir
    render(base_scene, os.path.join(out_dir, "base_cornell.png"), args.width, args.height, args.spp, max(args.depth, 4), args.seed, args.mis)
    for spp in (16, 64, 256):
        render(base_scene, os.path.join(out_dir, f"spp_{spp}.png"), args.width, args.height, max(1, min(spp, args.max_suite_spp)), max(args.depth, 4), args.seed + spp, args.mis)
    for depth in (2, 4, 6):
        render(base_scene, os.path.join(out_dir, f"depth_{depth}.png"), args.width, args.height, args.spp, depth, args.seed + depth, args.mis)
    render(environment_scene, os.path.join(out_dir, "environment_light.png"), args.width, args.height, args.spp, max(args.depth, 4), args.seed + 301, args.mis)
    render(mesh_light_scene, os.path.join(out_dir, "mesh_light.png"), args.width, args.height, args.spp, max(args.depth, 4), args.seed + 401, args.mis)
    render(mis_scene, os.path.join(out_dir, "mis_off.png"), args.width, args.height, args.spp, max(args.depth, 4), args.seed + 501, False)
    render(mis_scene, os.path.join(out_dir, "mis_on.png"), args.width, args.height, args.spp, max(args.depth, 4), args.seed + 501, True)


def parse_args():
    parser = argparse.ArgumentParser(description="Path tracer Monte Carlo para o Projeto 2 de INF2608.")
    parser.add_argument("--scene", choices=sorted(SCENES.keys()) + ["all"], default="base_cornell")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="Trabalho 2/base_cornell.png")
    parser.add_argument("--output-dir", default="Trabalho 2/img")
    parser.add_argument("--mis", action="store_true")
    parser.add_argument("--max-suite-spp", type=int, default=32, help="Limita custo da suite mantendo os nomes comparativos.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scene == "all":
        render_suite(args)
        return
    render(SCENES[args.scene], args.output, args.width, args.height, args.spp, args.depth, args.seed, args.mis)


if __name__ == "__main__":
    main()
