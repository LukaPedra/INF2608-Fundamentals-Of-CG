import numpy as np
from PIL import Image
import math
import sys
import time

# ==========================================
# UTILITÁRIOS VETORIAIS
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def reflect(I, N):
    """Reflexão do vetor I em relação à normal N."""
    return I - 2.0 * np.dot(I, N) * N

def refract(I, N, ior):
    """Lei de Snell: calcula a direção refratada. Retorna zeros em reflexão interna total."""
    cosi = np.clip(np.dot(I, N), -1.0, 1.0)
    etai, etat = 1.0, ior
    n = N.copy()
    if cosi < 0:
        cosi = -cosi          # raio entrando no meio
    else:
        etai, etat = etat, etai   # raio saindo do meio
        n = -N
    eta = etai / etat
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0:
        return np.zeros(3)    # Reflexão Interna Total
    return eta * I + (eta * cosi - math.sqrt(k)) * n

def fresnel(I, N, ior):
    """
    Calcula o coeficiente de Fresnel para um dielétrico (fração de luz refletida).
    Retorna kr ∈ [0, 1]. A fração transmitida é (1 - kr).
    """
    cosi = np.clip(np.dot(I, N), -1.0, 1.0)
    etai, etat = 1.0, ior
    if cosi > 0:              # raio saindo do meio — inverte os índices
        etai, etat = etat, etai
    sint = etai / etat * math.sqrt(max(0.0, 1.0 - cosi * cosi))
    if sint >= 1.0:
        return 1.0            # Reflexão Interna Total
    cost = math.sqrt(max(0.0, 1.0 - sint * sint))
    cosi = abs(cosi)
    rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
    rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))
    return (rs * rs + rp * rp) / 2.0

# ==========================================
# ESTRUTURAS BÁSICAS
# ==========================================

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))

class Material:
    def __init__(self, color, diffuse=0.8, specular=0.2, shininess=50,
                 is_reflective=False, reflectivity=0.0,
                 is_refractive=False, ior=1.0):
        self.color = np.array(color, dtype=float)
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.is_reflective = is_reflective
        self.reflectivity = reflectivity
        self.is_refractive = is_refractive
        self.ior = ior

class Hit:
    def __init__(self, t, point, normal, material):
        self.t = float(t)
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.material = material

# ==========================================
# GEOMETRIA
# ==========================================

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4.0 * a * c
        if disc <= 0:
            return None
        sq = math.sqrt(disc)
        t = (-b - sq) / (2.0 * a)
        if t < 1e-4:
            t = (-b + sq) / (2.0 * a)
        if t < 1e-4:
            return None
        p = ray.origin + t * ray.direction
        n = normalize(p - self.center)
        return Hit(t, p, n, self.material)

class Box:
    def __init__(self, bmin, bmax, material):
        self.bmin = np.array(bmin, dtype=float)
        self.bmax = np.array(bmax, dtype=float)
        self.material = material

    def intersect(self, ray):
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dir = np.where(ray.direction != 0,
                               1.0 / ray.direction,
                               np.sign(ray.direction) * 1e30)
        t0 = (self.bmin - ray.origin) * inv_dir
        t1 = (self.bmax - ray.origin) * inv_dir
        tmin = np.minimum(t0, t1).max()
        tmax = np.maximum(t0, t1).min()
        if tmin >= tmax:
            return None
        t = tmin if tmin > 1e-4 else tmax
        if t <= 1e-4:
            return None
        p = ray.origin + t * ray.direction
        n = self._get_normal(p)
        return Hit(t, p, n, self.material)

    def _get_normal(self, p):
        """Normal robusta: identifica a face pelo eixo dominante."""
        center = (self.bmin + self.bmax) * 0.5
        half   = (self.bmax - self.bmin) * 0.5
        d = p - center
        # A face atingida é aquela cujo gap (half - |d|) é mínimo
        gaps = half - np.abs(d)
        axis = int(np.argmin(gaps))
        n = np.zeros(3)
        n[axis] = 1.0 if d[axis] > 0 else -1.0
        return n

# ==========================================
# CENA (suporta múltiplas fontes de luz pontual)
# ==========================================

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = [
            {'pos': np.array([0.0, 9.0, 0.0]), 'color': np.array([1.0, 1.0, 1.0])}
        ]
        self.ambient = np.array([0.1, 0.1, 0.1])

    def add(self, obj):
        self.objects.append(obj)

    def add_light(self, pos, color=(1.0, 1.0, 1.0)):
        self.lights.append({
            'pos':   np.array(pos,   dtype=float),
            'color': np.array(color, dtype=float)
        })

    def closest_intersection(self, ray):
        closest, min_t = None, float('inf')
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit.t < min_t:
                min_t, closest = hit.t, hit
        return closest

# ==========================================
# MOTOR DE TRAÇADO DE RAIOS
# ==========================================

def trace(ray, scene, depth=0):
    if depth > 5:
        return np.zeros(3)

    hit = scene.closest_intersection(ray)
    if hit is None:
        return np.zeros(3)          # fundo preto

    mat    = hit.material
    point  = hit.point
    normal = hit.normal

    # ------------------------------------------------------------------
    # Material dielétrico (refratário): mistura Fresnel entre reflexão e
    # refração. Ignora iluminação Phong direta (material transparente).
    # ------------------------------------------------------------------
    if mat.is_refractive:
        kr      = fresnel(ray.direction, normal, mat.ior)
        outside = np.dot(ray.direction, normal) < 0
        bias    = normal * 1e-4

        # Componente refratada (transmitida)
        refr_color = np.zeros(3)
        if kr < 1.0:               # sem reflexão interna total
            refr_dir = refract(ray.direction, normal, mat.ior)
            if np.linalg.norm(refr_dir) > 1e-6:
                refr_origin = point - bias if outside else point + bias
                refr_color  = trace(Ray(refr_origin, refr_dir), scene, depth + 1)

        # Componente refletida
        refl_dir    = reflect(ray.direction, normal)
        refl_origin = point + bias if outside else point - bias
        refl_color  = trace(Ray(refl_origin, refl_dir), scene, depth + 1)

        return np.clip(refl_color * kr + refr_color * (1.0 - kr), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Material opaco: iluminação de Phong com múltiplas luzes e sombras
    # ------------------------------------------------------------------
    final_color = mat.color * scene.ambient

    for light in scene.lights:
        l_pos   = light['pos']
        l_color = light['color']
        l_vec   = l_pos - point
        l_dist  = np.linalg.norm(l_vec)
        l_dir   = l_vec / l_dist

        # Raio de sombra: objetos refratários (vidro) não bloqueiam totalmente a luz
        shadow_ray = Ray(point + normal * 1e-4, l_dir)
        shadow_hit = scene.closest_intersection(shadow_ray)
        in_shadow  = (shadow_hit is not None
                      and shadow_hit.t < l_dist
                      and not shadow_hit.material.is_refractive)

        if not in_shadow:
            # Componente difusa (Lambert)
            n_dot_l = max(0.0, np.dot(normal, l_dir))
            diffuse = mat.diffuse * mat.color * l_color * n_dot_l

            # Componente especular (Phong)
            v_dir   = normalize(-ray.direction)
            r_dir   = reflect(-l_dir, normal)
            r_dot_v = max(0.0, np.dot(r_dir, v_dir))
            specular = mat.specular * l_color * (r_dot_v ** mat.shininess)

            final_color += diffuse + specular

    # Reflexão para materiais opacos reflexivos (ex.: espelho)
    if mat.is_reflective:
        refl_dir   = reflect(ray.direction, normal)
        refl_ray   = Ray(point + normal * 1e-4, refl_dir)
        refl_color = trace(refl_ray, scene, depth + 1)
        final_color = (final_color * (1.0 - mat.reflectivity)
                       + refl_color * mat.reflectivity)

    return np.clip(final_color, 0.0, 1.0)

# ==========================================
# MATERIAIS
# ==========================================

mat_red          = Material([0.8, 0.1, 0.1])
mat_green        = Material([0.1, 0.8, 0.1])
mat_white        = Material([0.9, 0.9, 0.9])
mat_blue         = Material([0.1, 0.1, 0.8], diffuse=0.9, specular=0.1)
mat_mirror       = Material([0.9, 0.9, 0.9], diffuse=0.0, specular=0.0,
                             is_reflective=True, reflectivity=0.95)
mat_glass        = Material([1.0, 1.0, 1.0], is_refractive=True, ior=1.5)
mat_shiny_yellow = Material([0.8, 0.8, 0.1], diffuse=0.5, specular=0.8, shininess=100)
mat_matte_yellow = Material([0.8, 0.8, 0.1], diffuse=0.9, specular=0.1, shininess=5)

# ==========================================
# CENÁRIOS DE TESTE
# ==========================================

def build_cornell_box(scene):
    """Paredes da Cornell Box: chão, teto, fundo, esquerda (vermelha), direita (verde)."""
    scene.add(Box([-5,    0,    -5  ], [ 5,   0.1,  5], mat_white))  # Chão
    scene.add(Box([-5,    9.9,  -5  ], [ 5,  10.0,  5], mat_white))  # Teto
    scene.add(Box([-5,    0,    -5.1], [ 5,  10.0, -5], mat_white))  # Fundo
    scene.add(Box([-5.1,  0,    -5  ], [-5,  10.0,  5], mat_red))    # Esquerda
    scene.add(Box([ 5,    0,    -5  ], [ 5.1,10.0,  5], mat_green))  # Direita

def test_geometry(scene):
    """Teste 1.1 — Geometrias: interseção com Esferas e Caixas (método Slabs)."""
    build_cornell_box(scene)
    scene.add(Box([-1.5, 0.1, -1.5], [-0.5, 6.0, -0.5], mat_white))
    scene.add(Sphere([1.5, 4.0, -1.0], 1.5, mat_blue))

def test_phong(scene):
    """Teste 1.2 — Modelo de Phong com duas fontes de luz pontual."""
    build_cornell_box(scene)
    scene.lights[0]['pos'] = np.array([4.0, 8.0, 4.0])
    scene.add_light([-4.0, 8.0, 4.0], color=[0.6, 0.6, 1.0])   # segunda luz azulada
    scene.add(Sphere([-3, 1.5, -2], 1.5, mat_blue))
    scene.add(Sphere([ 0, 1.5, -2], 1.5, mat_shiny_yellow))
    scene.add(Sphere([ 3, 1.5, -2], 1.5, mat_matte_yellow))

def test_shadows(scene):
    """Teste 1.3 — Sombra dura: validação de oclusão e ausência de shadow acne."""
    build_cornell_box(scene)
    scene.lights[0]['pos'] = np.array([0.0, 8.0, 8.0])
    scene.add(Sphere([0, 3.0, -2], 2.5, mat_white))

def test_antialiasing(scene):
    """Teste 1.4 — Esfera de alto contraste para comparar 1 vs 16 amostras."""
    build_cornell_box(scene)
    scene.add(Sphere([0, 3.0, -2], 2.5, mat_red))

def test_reflection(scene):
    """Teste 2.1 — Material reflexivo (espelho); reflexão recursiva."""
    build_cornell_box(scene)
    scene.add(Sphere([0,   3.0, -2], 2.0, mat_mirror))
    scene.add(Sphere([2.5, 1.0,  0], 1.0, mat_red))

def test_refraction(scene):
    """Teste 2.2 — Material dielétrico (vidro, IOR=1.5) com equações de Fresnel."""
    build_cornell_box(scene)
    scene.add(Box([-2.0, 0.1, -4.5], [2.0, 8.0, -4.0], mat_red))  # pilar ao fundo
    scene.add(Sphere([0, 3.0, -1], 2.0, mat_glass))                # esfera de vidro

# ==========================================
# RENDERIZADOR PRINCIPAL
# ==========================================

def render(scenario_func, filename, width=400, height=400, samples=4):
    """
    Renderiza uma cena com amostragem em grade uniforme (n×n por pixel).
    O parâmetro 'samples' define o número de amostras; usa-se n = floor(√samples)
    por eixo, resultando em n² amostras totais (grade uniforme).
    """
    FOV    = 60.0
    aspect = width / height
    tan_h  = math.tan(math.radians(FOV / 2.0))

    camera_pos = np.array([0.0, 5.0, 8.0])

    scene = Scene()
    scenario_func(scene)

    # Grade uniforme: n amostras por eixo → n² amostras por pixel
    n_grid        = max(1, int(math.sqrt(samples)))
    total_samples = n_grid * n_grid
    image_data    = np.zeros((height, width, 3))

    print(f"\nRenderizando '{filename}' ({width}×{height}, "
          f"{total_samples} amostras/pixel, grade {n_grid}×{n_grid})...")
    start = time.time()

    for y in range(height):
        if y % 20 == 0:
            print(f"  Linha {y}/{height}")
        for x in range(width):
            pixel_color = np.zeros(3)

            # Distribuição uniforme: centro de cada célula da subgrade
            for sy in range(n_grid):
                for sx in range(n_grid):
                    u = x + (sx + 0.5) / n_grid
                    v = y + (sy + 0.5) / n_grid
                    ndc_x = ( 2.0 * u / width  - 1.0) * tan_h * aspect
                    ndc_y = (-2.0 * v / height + 1.0) * tan_h
                    ray   = Ray(camera_pos, np.array([ndc_x, ndc_y, -1.0]))
                    pixel_color += trace(ray, scene, depth=0)

            pixel_color /= total_samples
            # Correção gamma 2.2
            image_data[y, x] = np.power(np.clip(pixel_color, 0, 1), 1.0 / 2.2) * 255.0

    img = Image.fromarray(image_data.astype(np.uint8))
    img.save(filename)
    print(f"Salvo: '{filename}' ({time.time() - start:.2f}s)")

# ==========================================
# MENU DE EXECUÇÃO
# ==========================================

if __name__ == "__main__":
    print("=== Motor de Traçado de Raios — INF2608 ===")

    if len(sys.argv) < 2:
        print("Uso: python numpy_raytracer.py [código]")
        print("Códigos disponíveis:")
        print("  1.1  — Geometria (Esferas e Caixas)")
        print("  1.2  — Iluminação Phong (múltiplas luzes)")
        print("  1.3  — Sombras")
        print("  1.4a — Antialiasing OFF (1 amostra)")
        print("  1.4b — Antialiasing ON  (16 amostras)")
        print("  2.1  — Extensão: Reflexão")
        print("  2.2  — Extensão: Refração (Fresnel)")
        print("  all  — Todos em sequência")
        sys.exit()

    W, H = 400, 400
    cenarios = {
        "1.1":  (test_geometry,     "teste_1.1_geometria.png",  4),
        "1.2":  (test_phong,        "teste_1.2_phong.png",      4),
        "1.3":  (test_shadows,      "teste_1.3_sombras.png",    4),
        "1.4a": (test_antialiasing, "teste_1.4_aa_OFF.png",     1),
        "1.4b": (test_antialiasing, "teste_1.4_aa_ON.png",      16),
        "2.1":  (test_reflection,   "teste_2.1_reflexao.png",   4),
        "2.2":  (test_refraction,   "teste_2.2_refracao.png",   4),
    }

    escolha = sys.argv[1]
    if escolha == "all":
        for k, (func, fname, samp) in cenarios.items():
            render(func, fname, W, H, samp)
    elif escolha in cenarios:
        func, fname, samp = cenarios[escolha]
        render(func, fname, W, H, samp)
    else:
        print(f"Cenário '{escolha}' não encontrado.")
