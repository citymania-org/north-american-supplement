import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import spectra

import grf

EPS = 1e-7
CAM_NORM = (6**.5 / 4., 6**.5 / 4., .5)
LIGHT_NORM = (-6**.5 / 4., 6**.5 / 4., .5)
SHADE_STRENGTH = 128


def d3_to_cam(x, y, z):
    return (y - x, (y + x) / 2 - z)


class Face:
    next_id = 0

    def __init__(self, vertices, uv, group_id = None):
        self.id = self.__class__.next_id
        self.__class__.next_id += 1

        self.group_id = group_id
        self.vertices = vertices
        self.cam = np.array([d3_to_cam(x, y, z) for x, y, z in self.vertices])
        self.uv = uv
        self.normal = None

        if len(self.vertices) >= 3:
            # Newell's method of calculating normals
            pv = self.vertices[-1]
            n = np.array((0., 0., 0.))
            for v in self.vertices:
                n[0] += (pv[1] - v[1]) * (v[2] + pv[2])
                n[1] += (pv[2] - v[2]) * (v[0] + pv[0])
                n[2] += (pv[0] - v[0]) * (v[1] + pv[1])
                pv = v
            nl = np.linalg.norm(n)
            if -EPS < nl < EPS:
                print('No normal', self.vertices)
            else:
                self.normal = n / nl

    @property
    def direction(self):
        if self.normal is None:
            return 'f'
        for i, d in enumerate('xyz'):
            if -EPS < self.normal[i] - 1 < EPS:
                return d
        return 'f'

    def copy(self):
        return Face(self.vertices.copy(), self.uv.copy())

    def scale(self, mult):
        nvertices = mult * self.vertices
        return Face(nvertices, self.uv.copy())

    def rotate(self, matrix):
        vertices = self.vertices @ matrix.T
        points = vertices @ D3_TO_CAM.T
        return Face(vertices, self.uv.copy())
        # f.normal = matrix @ self.normal

    def contains_cam(self, cx, cy):
        px, py = self.cam[-1]
        nodes_left = 0
        for x, y in self.cam:
            if min(y, py) < cy and max(y, py) >= cy:
                f = (cy - y) / (py - y)
                if x + f * (px - x) < cx:
                    nodes_left += 1
            px, py = x, y
        return nodes_left % 2 == 1


    def iter_pixels_uv(self, w, h):
        # uv = self.uv * np.array((w + 2, - 5 * (h + 2))) + np.array((0, 5 * (h + 2)))
        uv = self.uv * np.array((w, h))
        minv = np.min(uv[:, 1], initial=1000) - .5
        maxv = np.max(uv[:, 1], initial=0) - .5
        # print(self.points)
        # print(miny, maxy, math.ceil(miny), math.floor(maxy) + 1)
        for cvi in range(math.ceil(minv), math.floor(maxv) + 1):
            cv = cvi + .5
            nodes = []
            puv, pc = uv[-1], self.vertices[-1]
            pu, pv = uv[-1]
            for i, (u, v) in enumerate(uv):
                c = self.vertices[i]
                if min(v, pv) < cv and max(v, pv) >= cv:
                    f = (cv - v) / (pv - v)
                    nodes.append((u + f * (pu - u),
                                  c + f * (pc - c)))
                pu, pv, pc = u, v, c

            nodes.sort(key=lambda x: x[0])

            for i in range(0, len(nodes), 2):
                ui, xyzi = nodes[i]
                uj, xyzj = nodes[i + 1]
                du = uj - ui
                if du < EPS:
                    continue
                f = 1. / du
                minu = math.ceil(ui - .5)
                maxu = math.floor(uj - .5)
                for u in range(minu, maxu + 1):
                    xyz = xyzi + (u - ui + .5) * f * (xyzj - xyzi)
                    yield (u, cvi, xyz)

    def iter_pixels_cam(self):
        miny = np.min(self.cam[:, 1], initial=1000)
        maxy = np.max(self.cam[:, 1], initial=0)
        # print(self.points)
        # print(miny, maxy, math.ceil(miny), math.floor(maxy) + 1)
        for cy in range(math.ceil(miny), math.floor(maxy) + 1):
            nodes = []
            puv, pc = self.uv[-1], self.vertices[-1]
            px, py = self.cam[-1]
            for i, uv in enumerate(self.uv):
                c = self.vertices[i]
                x, y = self.cam[i]
                if min(y, py) < cy and max(y, py) >= cy:
                    f = (cy - y) / (py - y)
                    nodes.append((x + f * (px - x),
                                  uv + f * (puv - uv),
                                  c + f * (pc - c)))
                px, py, puv, pc = x, y, uv, c

            nodes.sort(key=lambda x: x[0])

            for i in range(0, len(nodes), 2):
                xi, uvi, ci = nodes[i]
                xj, uvj, cj = nodes[i + 1]
                dx = xj - xi
                if dx < EPS:
                    continue
                f = 1. / dx
                minx = math.ceil(xi + .5)
                maxx = math.floor(xj + .5)
                for x in range(minx, maxx + 1):
                    c = ci + (x - xi - .5) * f * (cj - ci)
                    uv = uvi + (x - xi - .5) * f * (uvj - uvi)
                    yield (x, cy, uv, c)


class Geometry:
    def __init__(self, vertices, groups, texture_path):
        self.vertices = vertices
        self.groups = groups
        self.texture_path = texture_path

    @property
    def faces(self):
        return (f for faces in self.groups.values() for f in faces)

    @classmethod
    def import_mtl(cls, path):
        with open(path, 'r') as file:
            for l in file:
                ll = l.strip().split(' ', 1)
                if len(ll) != 2:
                    continue
                cmd, data = ll
                if cmd == 'newmtl':
                    print('TODO handle newmtl')
                elif cmd == 'map_Kd':
                    return path.parent.joinpath(data)
        return None

    @classmethod
    def import_obj(cls, path):
        Face.next_id = 0
        faces = []
        v = [None]
        groups = {}
        group_id = None
        vt = [None]
        mtllib = None
        with open(path, 'r') as file:
            for l in file:
                ls = l.strip().split(' ')
                if ls[0] == 'mtllib':
                    mtllib = ls[1]
                elif ls[0] == 'usemtl':
                    print('TODO handle usemtl')
                elif ls[0] == 'v':
                    x, z, yb = (float(x) / 2 for x in ls[1:])
                    v.append((x, -yb, z))
                elif ls[0] == 'vt':
                    tu, tv = tuple(float(x) for x in ls[1:])
                    # tu = tu * (w + 2)
                    # tv = (1 - tv) * 5 * (h + 2)
                    vt.append((tu, 1 - tv))
                # elif ls[0] == 'vn':
                #     nv = tuple(float(x) for x in ls[1:])
                elif ls[0] == 'f':
                    face_d3 = []
                    face_uv = []
                    face_v = []
                    for p in ls[1:]:
                        ps = p.split('/')
                        vi = int(ps[0])
                        vti = int(ps[1])
                        face_v.append(vi)
                        face_d3.append(v[vi])
                        face_uv.append(vt[vti])
                    f = Face(np.array(face_d3), np.array(face_uv))
                    f.vertex_ids = face_v
                    faces.append(f)
                    groups.setdefault(group_id, []).append(f)
                    f.group_id = group_id
                elif ls[0] == 's':
                    group_id = ls[1].strip()
                    if group_id in ('off', '0'):
                        group_id = None
        # except FileNotFoundError:
        #     print(f'File not found: {filename}')

        assert mtllib is not None
        texpath = cls.import_mtl(path.parent.joinpath(mtllib))

        res = cls(v, groups, texpath)
        return res


def generate_d3_points(geometry):
    d3_list = []
    texture = Image.open(geometry.texture_path)
    texture = texture.convert('RGBA')
    w, h = texture.size
    nptex = np.array(texture)
    nptex = nptex.view(dtype=np.uint32).reshape(nptex.shape[:-1])
    for f in geometry.faces:
        if len(f.vertices) == 0:
            print('Face has no vertices')
            continue

        if f.normal is None:
            print('Face has no normal!', f.vertices)
            continue

        if -EPS < np.dot(f.normal, f.normal) < EPS:
            print('Normal is 0', f.normal, f.vertices)
            continue

        vl = []
        cl = []
        for u, v, xyz in f.iter_pixels_uv(w, h):
            cl.append(nptex[v, u])
            vl.append(xyz)

        if vl:
            d3_list.append((f, np.array(vl), cl))
        else:
            d3_list.append((f, np.empty((0, 3)), np.empty((0,))))

    return d3_list


def render_upscale(points, zoom=1, noise=1.5):
    sg_vlr = defaultdict(list)
    sg_cl = defaultdict(list)
    z = {}
    for f, vl, cl in points:
        fr = f
        if np.dot(fr.normal, CAM_NORM) < 0:
            continue

        vlr = vl
        sgkey = f.group_id
        if sgkey is None: sgkey = id(f)
        sg_vlr[sgkey].append(vlr)
        sg_cl[sgkey].append(np.array(cl, dtype=np.uint32))

        if zoom > 1:
            fr = fr.scale(zoom)

        for x, y, uv, c in fr.iter_pixels_cam():
            p = (x, y)
            x = z.get(p)
            if zoom > 1:
                c = c / zoom
            if x is not None and x[0][2] >= c[2]:
                continue
            z[p] = (c, sgkey, f)

    if not bool(z):
        print('No z')
        return

    minx, maxx = min(p[0] for p in z), max(p[0] for p in z)
    miny, maxy = min(p[1] for p in z), max(p[1] for p in z)
    npimg = np.zeros((maxy - miny + 1, maxx - minx + 1), dtype=np.uint32)

    sge = set()
    for k in sg_vlr:
        if len(sg_vlr[k]) <= 0:
            continue

        if len(sg_vlr[k]) == 1:
            sg_vlr[k] = sg_vlr[k][0]
            sg_cl[k] = sg_cl[k][0]
            continue
        sge.add(k)
        sg_vlr[k] = np.concatenate(sg_vlr[k])
        sg_cl[k] = np.concatenate(sg_cl[k])

    # for k in sg_vlr:
    #     print('COLOUR', sg_cl[k])

    # for p, (pcz, sgkey, f) in z.items():
    #     ps[p[0] - minx, p[1] - miny] = face_layers[id(f)]
    # ps.close()
    # return s

    shade_cache = {}
    # for p, (pcz, vlr, cl, f) in z.items():
    for p, (xyz, sgkey, f) in z.items():
        # if sgkey not in sge:
        #     ps[p[0] - minx, p[1] - miny] = 0xb8
        #     continue
        # ps[p[0] - minx, p[1] - miny] = (sgkey + 1) % 256 if sgkey in sge else 0
        # continue
        # find nearest point
        if len(sg_vlr[sgkey]) <= 0:
            continue

        # ci = np.argmin(np.linalg.norm(sg_vlr[sgkey] - xyz, axis=1))
        ci = np.argmin(np.linalg.norm(sg_vlr[sgkey] - xyz, axis=1) * (np.random.rand(len(sg_vlr[sgkey])) * noise + 1))
        colour = sg_cl[sgkey][ci]
        if not colour:
            continue
        shade_key = (f, colour)
        shaded_colour = shade_cache.get(shade_key)
        if shaded_colour is None:
            light = max(np.dot(f.normal, LIGHT_NORM), 0)
            b = (colour >> 16) & 0xFF
            g = (colour >> 8) & 0xFF
            r = colour & 0xFF
            # shade = int(255 - SHADE_STRENGTH * (1 - light))
            # r = (r * shade) // 255
            # g = (g * shade) // 255
            # b = (b * shade) // 255
            # shaded_colour = (b << 16) | (g << 8) | r
            # shaded_colour = colour
            # shade_cache[shade_key] = shaded_colour

            colour = spectra.rgb(r / 255., g / 255., b / 255.)
            colour = colour.darken((1 - light) * SHADE_STRENGTH / 4)
            r, g, b = colour.clamped_rgb
            shaded_colour = int(r * 255) | (int(g * 255) << 8) | (int(b * 255) << 16)
            shade_cache[shade_key] = shaded_colour

        # ps[p[0] - minx, p[1] - miny] = PALETTE[colour]
        npimg[p[1] - miny, p[0] - minx] = shaded_colour | 0xFF000000

    im = Image.fromarray(npimg, mode='RGBA')
    # im.putpalette(grf.PALETTE)
    return minx, miny, im


def debug_sprites(sprites, scale):
    rx, ry = 0, 0

    for s, sscale in sprites:
        rx += s.w * sscale
        ry = max(ry, s.h * sscale)
    im = Image.new('RGBA', (rx + 10 * len(sprites) - 10, ry))
    x = 0
    for s, sscale in sprites:
        simg = s.get_image()[0]
        simg = simg.resize((simg.size[0] * sscale, simg.size[1] * sscale), Image.NEAREST)
        im.paste(simg, (x, 0))
        x += s.w * sscale + 10
    im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.NEAREST)
    im.show()


class House(grf.SpriteGenerator):
    def __init__(self, id, path):
        self.id = id
        self.path = Path(path)
        self.geometry = Geometry.import_obj(self.path)
        self.points = generate_d3_points(self.geometry)
        self.origin = d3_to_cam(0, 0, 0)

    def debug_sprites(self):
        r = self.get_sprites(None)
        x1, x2, x4 = r[-1].sprites
        debug_sprites((
            (x1, 4),
            (x2, 2),
            (x4, 1)),
            10
        )

    def get_sprites(self, g):
        res = [grf.ReplaceOldSprites([(self.id, 1)])]
        ox1, oy1, x1 = render_upscale(self.points, zoom=1, noise=1.5)
        ox2, oy2, x2 = render_upscale(self.points, zoom=2, noise=1.5)
        ox4, oy4, x4 = render_upscale(self.points, zoom=4, noise=1.5)
        ox, oy = 0, -1
        res.append(grf.AlternativeSprites(
            grf.ImageSprite(x1, zoom=grf.ZOOM_4X, xofs=ox + ox1, yofs=oy + oy1),
            grf.ImageSprite(x2, zoom=grf.ZOOM_2X, xofs=ox * 2 + ox2 + 1, yofs=oy * 2 + oy2 + 1),
            grf.ImageSprite(x4, zoom=grf.ZOOM_NORMAL, xofs=ox * 4 + ox4 + 2, yofs=oy * 4 + oy4 + 2),
        ))
        return res
