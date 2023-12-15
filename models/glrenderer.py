# code adapeted from frankmocap https://github.com/facebookresearch/frankmocap
import numpy as np
import cv2
import os
import trimesh
from os.path import join
from sklearn.preprocessing import normalize
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
if __name__ == "__main__":
    from render_util import ComputeNormal, loadShader, createProgram
else:
    from .render_util import ComputeNormal, loadShader, createProgram

_glut_window = None

class glRenderer(object):
    def __init__(self, data_path, width=640, height=480, render_mode="geo", antialiasing=True):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.antialiasing = antialiasing

        self.display_mode = GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE

        self.program_files = dict()
        self.program_files['color'] = [join(data_path, 'simple140.fs'), join(data_path, 'simple140.vs')]
        self.program_files['normal'] = [join(data_path, 'normal140.fs'), join(data_path, 'normal140.vs')]
        self.program_files['geo'] = [join(data_path, 'colorgeo140.fs'), join(data_path, 'colorgeo140.vs')]
        self.program_files['flat'] = [join(data_path, 'flatgeo140.fs'), join(data_path, 'flatgeo140.vs')]
        self.program_files['test'] = [join(data_path, 'test.fs'), join(data_path, 'test.vs')]

        global _glut_window
        if _glut_window is None:
            glutInit()
            glutInitDisplayMode(self.display_mode)
            glutInitWindowSize(self.width, self.height)
            glutInitWindowPosition(0, 0)
            _glut_window = glutCreateWindow("GL_Renderer")

            glEnable(GL_DEPTH_CLAMP)
            glEnable(GL_DEPTH_TEST)

            glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE)
            glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE)
            glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE)

            glViewport(0, 0, self.width, self.height)
        
        else:
            print('created')

        self.program = None
        # print(self.program_files[self.render_mode])
        self.initShaderProgram(self.program_files[self.render_mode])

        # Init Uniform variables
        self.model_mat_unif = glGetUniformLocation(self.program, 'ModelMat')
        self.persp_mat_unif = glGetUniformLocation(self.program, 'PerspMat')

        self.vertex_buffer = glGenBuffers(1)
        self.color_buffer = glGenBuffers(1)
        self.normal_buffer = glGenBuffers(1)
        self.index_buffer = glGenBuffers(1)     #for Mesh face indices. Without this vertices should be repeated and ordered (3x times bigger)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)       #So that texture doesnt have to be power of 2
        self.backgroundTextureID = glGenTextures(1)

    def initShaderProgram(self, program_files):
         # Init shader programs
        shader_list = []
        for program_file in program_files:
            _, ext = os.path.splitext(program_file)
            if ext == '.vs':
                shader_list.append(loadShader(GL_VERTEX_SHADER, program_file))
            elif ext == '.fs':
                shader_list.append(loadShader(GL_FRAGMENT_SHADER, program_file))
            elif ext == '.gs':
                shader_list.append(loadShader(GL_GEOMETRY_SHADER, program_file))

        if self.program is not None:
            glDeleteProgram(self.program)

        self.program = createProgram(shader_list)
        for shader in shader_list:
            glDeleteShader(shader)

    def set_window(self, height, width):
        if height != self.height or width != self.width:
            self.height = height
            self.width = width
        glutReshapeWindow(self.width,self.height)
        # glViewport(0, 0, self.width, self.height)

    def setCameraViewOrth(self):
        # glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()

        texHeight,texWidth =  self.height, self.width
        texHeight*=0.5
        texWidth*=0.5
        glOrtho(-texWidth, texWidth, -texHeight, texHeight, -1500, 1500)
        # glMatrixMode(GL_MODELVIEW)
        # glLoadIdentity()
        gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

    def drawBackgroundOrth(self, img):
        '''
        use two triangles to draw the background
        '''
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, self.backgroundTextureID)
        texHeight, texWidth = img.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, img.data)

        texHeight*=0.5
        texWidth*=0.5

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_QUADS)
        # glColor3f(1.0, 1.0, 1.0)

        d = 10.

        glTexCoord2f(0, 0)
        P = np.array([-texWidth, -texHeight, d])
        glVertex3f(P[0] , P[1] , P[2] );  # K^{-1} [0, 0, 1]^T
        glTexCoord2f(1, 0)
        P = [texWidth, -texHeight, d]
        glVertex3f(P[0] , P[1] , P[2] );  # K^{-1} [0, 0, 1]^T
        glTexCoord2f(1, 1)
        P = [texWidth, texHeight, d]
        glVertex3f(P[0] , P[1] , P[2] );  # K^{-1} [0, 0, 1]^T
        glTexCoord2f(0, 1)
        P = [-texWidth, texHeight, d]
        glVertex3f(P[0] , P[1] , P[2] );  # K^{-1} [0, 0, 1]^T
        glEnd()

        glEnable(GL_LIGHTING)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)

    def render(self, verts, faces, calib, persp, img=None, use_facenormals=False, draw_line=False):
        '''
        param: img [h w 3]
        param: verts [n 3]
        param: faces [f 3]
        '''
        assert verts.shape[1] == 3 and faces.shape[1] == 3
        if img is not None:
            h, w, _ = img.shape
            if h != self.height or w != self.width:
                self.set_window(h, w)
        else:
            h = self.height
            w = self.width

        normal = ComputeNormal(verts[None], faces, use_facenormals)[0].astype(np.float32)
        # normal = normal.transpose()
        # normal = np.vstack([normal, np.zeros_like(normal[0])])
        # normal = (calib@normal)[:3].transpose()
        # normal = normalize(normal)

        color = np.ones_like(normal)

        mesh_idx = faces.astype(np.int32).flatten()

        # verts = verts.transpose()
        # verts = np.vstack([verts, np.ones_like(verts[0])])
        # xyzw = persp@calib@verts
        # xyz = xyzw[:3]
        # xyz[1] = -xyz[1]

        glUseProgram(0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        if self.antialiasing:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glEnable(GL_POLYGON_SMOOTH)
            glEnable(GL_MULTISAMPLE)
        else:
            glDisable(GL_BLEND)
            glDisable(GL_MULTISAMPLE)

        if img is not None:
            self.setCameraViewOrth()
            self.drawBackgroundOrth(img.copy())

        glUseProgram(self.program)
        glClear(GL_DEPTH_BUFFER_BIT)
        # glShadeModel(GL_FLAT)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, verts.astype(np.float32).copy(), GL_STATIC_DRAW)
        # glBufferData(GL_ARRAY_BUFFER, xyz.transpose().copy(), GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glBufferData(GL_ARRAY_BUFFER, normal.copy(), GL_STATIC_DRAW)

        if self.render_mode == 'geo' or self.render_mode == 'flat':
            glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
            glBufferData(GL_ARRAY_BUFFER, color, GL_STATIC_DRAW)


        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh_idx, GL_STATIC_DRAW)

        # glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, np.eye(4, dtype=np.float32))
        # glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, np.eye(4, dtype=np.float32))

        glUniformMatrix4fv(self.model_mat_unif, 1, GL_TRUE, calib.astype(np.float32).copy())
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_TRUE, persp.astype(np.float32).copy())

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        glEnableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh_idx, GL_STATIC_DRAW)
        glDrawElements(GL_TRIANGLES, len(mesh_idx), GL_UNSIGNED_INT, None)

        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

        glutSwapBuffers()
        glutPostRedisplay()


        if draw_line:
            trans_verts = verts.transpose()
            trans_verts = np.vstack([trans_verts, np.ones_like(trans_verts[0])])
            xyzw = persp@calib@trans_verts
            if self.render_mode == 'geo':
                xyz = xyzw[:3] / xyzw[3:4]
            else:
                xyz = xyzw[:3]
            xyz[1] = -xyz[1]

            all_line = []
            for face in faces:
                tmp = sorted(face)
                all_line.append((tmp[0], tmp[1]))
                all_line.append((tmp[0], tmp[2]))
                all_line.append((tmp[1], tmp[2]))

            all_line = set(all_line)
            glLineWidth(1)
            for item in all_line:
                glBegin(GL_LINES)
                glColor3f(0.0, 0.0, 0.0)
                if self.render_mode == 'flat':
                    glVertex3f(xyz[0, item[0]], xyz[1, item[0]], -10)
                    glVertex3f(xyz[0, item[1]], xyz[1, item[1]], -10)
                    # glVertex2f((xyz[0, item[0]]+1)*self.width/2, (xyz[0, item[1]]+1)*self.height/2)
                    # glVertex2f((xyz[1, item[0]]+1)*self.width/2, (xyz[1, item[1]]+1)*self.height/2)
                else:
                    glVertex3f(xyz[0, item[0]], xyz[1, item[0]], xyz[2, item[0]])
                    glVertex3f(xyz[0, item[1]], xyz[1, item[1]], xyz[2, item[1]])

                glEnd()


            # for face in faces:
            #     glBegin(GL_LINE_LOOP)
            #     glColor3f(0.0, 0.0, 0.0)
            #     if self.render_mode == 'flat':
            #         glVertex3f(xyz[0, face[0]], xyz[1, face[0]], -10)
            #         glVertex3f(xyz[0, face[1]], xyz[1, face[1]], -10)
            #         glVertex3f(xyz[0, face[2]], xyz[1, face[2]], -10)
            #     else:
            #         glVertex3f(xyz[0, face[0]], xyz[1, face[0]], xyz[2, face[0]])
            #         glVertex3f(xyz[0, face[1]], xyz[1, face[1]], xyz[2, face[1]])
            #         glVertex3f(xyz[0, face[2]], xyz[1, face[2]], xyz[2, face[2]])

            #     glEnd()
            glFlush()


        glReadBuffer(GL_BACK)   #GL_BACK is Default in double buffering
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT, outputType=None)
        rgb = data.reshape(self.height, self.width, -1)
        rgb = np.flip(rgb, 0)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)    
        return bgr


if __name__ == "__main__":
    d2c = np.array([
    [ 9.99968867e-01, -6.80652917e-03, -9.22235761e-03, -5.44798585e-02], 
    [ 6.69376504e-03,  9.99922175e-01, -1.15625133e-02, -3.41685168e-04], 
    [ 9.27761963e-03,  1.14376287e-02,  9.99882174e-01, -2.50539462e-03], 
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00], ], dtype=np.float32)

    # color intrinsic
    c_fx = 1063.8987
    c_fy = 1063.6822
    c_cx = 954.1103
    c_cy = 553.2578

    from glob import glob
    import trimesh
    import random
    dires = glob(join('/home/llx/deephuman/dataset', '*/*'))

    res = 512
    factor = 1080 / res
    renderer = glRenderer('/home/llx/sdfgcn/data/shaders', res, res, render_mode="geo")

    persp = np.eye(4, dtype=np.float32)
    persp[0, 0] = c_fx / factor / (res//2)
    persp[1, 1] = c_fy / factor / (res//2)
    persp[2, 2] = 0.
    persp[2, 3] = -1.
    persp[3, 3] = 0.
    persp[3, 2] = 1.

    random.shuffle(dires)
    
    for dire in dires:
        img = cv2.imread(join(dire, 'color.jpg'))
        extra_data = np.load(join(dire, 'extra.npy'), allow_pickle=True).item()
        bbox = extra_data['bbox'].astype(np.int32)
        min_x, min_y, max_x, max_y = bbox
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, img.shape[1])
        max_y = min(max_y, img.shape[0])

        center_x = int((min_x+max_x)/2)
        h,w,_ = img.shape
        min_x = center_x - h//2
        max_x = center_x + h//2
        img = img[:, min_x:max_x]
        img = cv2.resize(img, (res, res))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mesh = trimesh.load(join(dire, 'mesh_watertight.obj'))
        verts = mesh.vertices.copy().astype(np.float32) # N 3
        faces = mesh.faces.copy().astype(np.int32)


        with open(join(dire, 'smpl_params.txt'), 'r') as fp:
            lines = fp.readlines()
            lines = [l.strip() for l in lines]

            root_mat_data = lines[3].split(' ') + lines[4].split(' ') +\
                lines[5].split(' ') + lines[6].split(' ')

            root_mat_data = filter(lambda s: len(s)!=0, root_mat_data)
            root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))

            root_rot = root_mat[:3, :3]
            root_tran = root_mat[:3, 3]

        
        cam = np.loadtxt(join(dire, 'cam.txt')).astype(np.float32)
        calib = np.eye(4, dtype=np.float32)
        calib[:3,:3] = root_rot
        calib[:3, 3] = root_tran
        calib = d2c @ cam @ calib

        persp[0, 2] = (c_cx-min_x) / factor / (res//2) - 1
        persp[1, 2] = c_cy / factor / (res//2) - 1

        ret = renderer.render(verts, faces, calib, persp, img=rgb)
        # ret = renderer.render(verts, faces, calib, persp)

        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2BGR)
        ret = (ret*255).astype(np.uint8)
        cv2.imwrite("test.jpg", ret)

        verts = verts.transpose()
        verts = np.vstack([verts, np.ones_like(verts[0])])
        xyzw = persp@calib@verts
        xyz = xyzw[:3] / xyzw[3:4]

        print(xyz[2].max())
        print(xyz[2].min())
        xy = xyzw[:2] / xyzw[3:4]
        xy = xy.transpose()
        x = (xy[:,0] * (res//2) + (res//2)).astype(np.int32)
        y = (xy[:,1] * (res//2) + (res//2)).astype(np.int32)

        img[y,x] = 255
        cv2.imwrite("t1.jpg", img)

        break

