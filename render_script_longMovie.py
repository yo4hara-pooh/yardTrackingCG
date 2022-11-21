import os
import math
from random import random
import bpy
from mathutils import Vector
from collections import defaultdict

def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


class Box:
    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return round(self.min_x * self.dim_x)

    @property
    def y(self):
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self):
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self):
        return round((self.max_y - self.min_y) * self.dim_y)

    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % \
               (self.x, self.y, self.width, self.height)

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.
    Negative 'z' value means the point is behind the camera.
    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.
    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """
    mat = cam_ob.matrix_world.normalized().inverted()
    me = me_ob.to_mesh(preserve_all_data_layers=True)
    me.transform(me_ob.matrix_world)
    me.transform(mat)
    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'
    lx = []
    ly = []
    for v in me.vertices:
        co_local = v.co
        z = -co_local.z
        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]
        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)
        ##ここではみ出るのを除外する
        lx.append(x)
        ly.append(y)
    min_x = min(lx)
    max_x = max(lx)
    min_y = min(ly)
    max_y = max(ly)
    ww = max_x - min_x
    hh = max_y - min_y

    min_x = clamp(min_x, 0.0, 1.0)
    max_x = clamp(max_x, 0.0, 1.0)
    min_y = clamp(min_y, 0.0, 1.0)
    max_y = clamp(max_y, 0.0, 1.0)
    ww_clamped = max_x - min_x
    hh_clamped = max_y - min_y    
    disp_area = (ww_clamped*hh_clamped)/(ww*hh) if ww*hh > 0.000001 else 0.0
    
    # bpy.data.meshes.remove(me)
    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac
    return (min_x, min_y, max_x, max_y, dim_x, dim_y, disp_area)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


# base_dir = "c:/tmp/blender/result"
trailer_headbox = 'headbox'
trailer_bodybox = 'containerbox'
trailer_head = 'head'
trailer_body = 'container'
#container = 'd02'
FRAME_NUM = 2000
SMALLEST_SIZE = 40
# enable
enable_gpus("CUDA")
# bpy.ops.render.engine = 'BLENDER_EEVEE' #CYCLES

print('----------------- Objects ------------')
cameras = []
for e in bpy.data.objects:
    if 'Camera' in e.name:
        print(e.name)
        cameras.append(e)
print('----------------- Objects END ------------\n')
CAM_NUM = min(9, len(cameras))
# CAM_NUM = 1
for id in range(CAM_NUM):
    bpy.context.scene.camera = cameras[id]

    base_dir = os.path.join(os.getcwd(), cameras[id].name + '/rendering_data_long')
    img_dir = os.path.join(base_dir, "images")
    ann_dir = os.path.join(base_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    # geometry_file_path = os.path.join(base_dir, "geometry.txt")
    # donut = bpy.data.collections['Truck']
    json_fn = f'annotation_all.json'
    json_path = os.path.join(ann_dir, json_fn)
    VGA_WIDTH, VGA_HEIGHT = 640, 640
    RENDER_WIDTH, RENDER_HEIGHT = 640, 360
    OBI_HALF_H = (VGA_HEIGHT - RENDER_HEIGHT) // 2
    MABIKI_FPS = 6  # 5fps

    with open(json_path, "w") as json_file:
        json_file.write('[\n')
        has_prev_frame = False
        # for i in range(0, FRAME_NUM, MABIKI_FPS):
        for i in range(FRAME_NUM):
            #        color = (random(), random(), random(), 1)
            #        donut.rotation_euler[0] = random() * math.pi
            #        donut.rotation_euler[1] = random() * math.pi
            #        donut.rotation_euler[2] = random() * math.pi
            #        donut.location.x = random()
            #        donut.location.y = random()
            #        donut.location.z = random()
            #        donut.active_material.node_tree.nodes['Principled BSDF'].inputs["Base Color"].default_value = color
            bpy.context.scene.frame_set(i)
            if i % MABIKI_FPS > 0: continue
            print("rendering frame [{}]-----------".format(i + 1))

            fn = f'image{i:05d}.png'
            path = os.path.join(img_dir, fn)
            bpy.context.scene.render.filepath = path

            bpy.ops.render.render(write_still=True)
            # label x y w h
            fn = f'image{i:05d}.txt'
            path = os.path.join(ann_dir, fn)

            # json_row_comment = f'#frame_id: {(i+1):05d}\n'
            # json_file.write(json_row_comment)
            if has_prev_frame: json_file.write(',\n')
            json_file.write('\t[')
            with open(path, "w") as file:
                has_before = False
                my_trailerhead = defaultdict(lambda: None)
                my_trailerheadbox = defaultdict(lambda: None)
                my_trailerbody = defaultdict(lambda: None)
                my_trailerbodybox = defaultdict(lambda: None)
                my_obj_dict = [my_trailerbody, my_trailerhead, my_trailerbodybox, my_trailerheadbox]
                #check visible area of trailers
                for e in bpy.data.objects:
                    # if "Truck" in e.name:
                    class_id = -1
                    
                    #elif trailer_body in e.name or container in e.name: class_id = 0                                        
                    #逆順はダメ！！！名前の被る関係があるので
                    if trailer_headbox in e.name: class_id = 3
                    elif trailer_bodybox in e.name: class_id = 2
                    elif trailer_head in e.name: class_id = 1
                    elif trailer_body in e.name: class_id = 0
                    if class_id < 0: continue
                    #real trailer not visible? continue
                    if class_id < 2 and e.hide_render: continue
                    obj_id = int(e.name[-3:])
                    x0, y0, x1, y1, _, _, disp_area = camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, e)
                    y1, y0 = 1 - y0, 1 - y1
                    # print(x0, y0, x1, y1)
                    # row = "%s,%i,%i,truck,%i,%i,%i,%i" % (f, b.dim_x, b.dim_y, b.x, b.y, b.x + b.width, b.y + b.height)
                    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
                    if w <= 0 or h <= 0 or w >= 0.75 or h >= 0.75: continue

                    not_visible = (class_id >= 2 and disp_area < 0.8) or (class_id < 2 and disp_area < 0.33)
                    if not_visible: continue
                    my_obj_dict[class_id][obj_id] = (cx, cy, w, h)

                #only accept both box and part visible objects
                good_head_key = my_trailerhead.keys() & my_trailerheadbox.keys()
                good_body_key = my_trailerbody.keys() & my_trailerbodybox.keys()
                ittaigata_key = good_head_key & good_body_key
                print('good_head_keys: ', good_head_key)
                for k in good_head_key:
                    class_id = 1
                    cx, cy, w, h = my_trailerheadbox[k]
                    json_cx = cx * RENDER_WIDTH
                    json_cy = cy * RENDER_HEIGHT + OBI_HALF_H
                    json_w = w * RENDER_WIDTH
                    json_h = h * RENDER_HEIGHT
                    json_lx = json_cx - json_w * 0.5
                    json_ly = json_cy - json_h * 0.5
                    #if json_w < SMALLEST_SIZE and json_w < json_h * 0.5: continue
                    #not annotate object less than 0.8

                    # print(class_id, cx, cy, w, h)
                    row = f'{class_id} {cx} {cy} {w} {h}\n'
                    file.write(row)

                    new_flg = 'true'
                    new_auto_flg = 'false'
                    overlap_flg = 'false'

                    # json出力       
                    json_prev_end = '\n'
                    if has_before: json_prev_end = ',\n'
                    json_file.write(json_prev_end)
                    json_row = '\t{\n\t\t' + '"class": {},\n\t\t"x": {},\n\t\t"y": {},\n\t\t"w": ' \
                                                '{},\n\t\t"h": {},\n\t\t"new": {},\n\t\t"new_auto": {},' \
                                                '\n\t\t"overlap": {}'.format(class_id, json_lx, json_ly, json_w,
                                                                            json_h, new_flg, new_auto_flg,
                                                                            overlap_flg) + '\n\t}'
                    json_file.write(json_row)
                    has_before = True
                print('ittaigata_key: ', ittaigata_key)
                for k in ittaigata_key:
                    cx0, cy0, w0, h0 = my_trailerheadbox[k]                    
                    cx1, cy1, w1, h1 = my_trailerbodybox[k]
                    lx0, ly0 = cx0 - w0*0.5, cy0 - h0*0.5
                    rx0, ry0 = cx0 + w0*0.5, cy0 + h0*0.5
                    lx1, ly1 = cx1 - w1*0.5, cy1 - h1*0.5                    
                    rx1, ry1 = cx1 + w1*0.5, cy1 + h1*0.5
                    lx, ly = min(lx0, lx1), min(ly0, ly1)
                    rx, ry = max(rx0, rx1), max(ry0, ry1)
                    ww, hh = rx - lx, ry - ly
                    cx, cy = lx + ww*0.5, ly + hh*0.5
                    class_id = 0

                    json_cx = cx * RENDER_WIDTH
                    json_cy = cy * RENDER_HEIGHT + OBI_HALF_H
                    json_w = ww * RENDER_WIDTH
                    json_h = hh * RENDER_HEIGHT
                    json_lx = json_cx - json_w * 0.5
                    json_ly = json_cy - json_h * 0.5
                    #if json_w < SMALLEST_SIZE and json_w < json_h * 0.5: continue
                    #not annotate object less than 0.8

                    # print(class_id, cx, cy, w, h)
                    row = f'{class_id} {cx} {cy} {ww} {hh}\n'
                    file.write(row)

                    new_flg = 'true'
                    new_auto_flg = 'false'
                    overlap_flg = 'false'

                    # json出力       
                    json_prev_end = '\n'
                    if has_before: json_prev_end = ',\n'
                    json_file.write(json_prev_end)
                    json_row = '\t{\n\t\t' + '"class": {},\n\t\t"x": {},\n\t\t"y": {},\n\t\t"w": ' \
                                                '{},\n\t\t"h": {},\n\t\t"new": {},\n\t\t"new_auto": {},' \
                                                '\n\t\t"overlap": {}'.format(class_id, json_lx, json_ly, json_w,
                                                                            json_h, new_flg, new_auto_flg,
                                                                            overlap_flg) + '\n\t}'
                    json_file.write(json_row)
                    has_before = True
            json_file.write('\n\t]')
            has_prev_frame = True
            # json_row_comment = f'   #frame_id: {(i+1):05d} END\n'
            # json_file.write(json_row_comment)

            json_file.flush()
        # close json
        json_file.write('\n]\n')
