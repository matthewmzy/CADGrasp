"""
Last modified date: 2023.07.04
Author: Jialiang Zhang
Description: observer class that converts state to observation
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh as tm
import torch
import transforms3d
from OpenGL.GL import *
from OpenGL.EGL import *
from OpenGL.EGL.EXT.device_base import egl_get_devices
from OpenGL.raw.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT
from OpenGL.GLU import *
from OpenGL.GL.shaders import *
import ctypes
import glfw
import glm
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'


def load_shaders():
    # read the shaders
    VertexShaderCode = """
        #version 330
        layout (location = 0) in vec3 Position;
        uniform mat4 MVP;
        uniform mat4 MV;
        out vec3 pos_camera;
        void main() {
            gl_Position = MVP * vec4(Position, 1.0);
            pos_camera = (MV * vec4(Position, 1.0)).xyz;
        }
    """
    FragmentShaderCode = """
        #version 330
        layout (location = 0) out vec4 color;
        in vec3 pos_camera;
        uniform int id;
        void main() {
            color.xyz = pos_camera;
            color.w = id;
        }
    """
    
    # compile shaders
    ProgramID = compileProgram(compileShader(VertexShaderCode, GL_VERTEX_SHADER), compileShader(FragmentShaderCode, GL_FRAGMENT_SHADER))
    
    return ProgramID


class Observer:
    def __init__(self, image_width, image_height, intrinsics, gpu=None, extrinsics=None):
        self.image_width = image_width
        self.image_height = image_height
        if gpu is not None:
            # 1. Initialize EGL
            eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_get_devices()[gpu], None)
            major = ctypes.c_uint32()
            minor = ctypes.c_uint32()
            eglInitialize(eglDpy, ctypes.byref(major), ctypes.byref(minor))
            # 2. Select an appropriate configuration
            config_attribs = [
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_BLUE_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_RED_SIZE, 8,
                EGL_DEPTH_SIZE, 8,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE
            ]
            config = EGLConfig()
            numConfigs = ctypes.c_uint32()
            eglChooseConfig(eglDpy, config_attribs, ctypes.byref(config), 1, ctypes.byref(numConfigs))
            # 3. Create a surface
            pbufferAttribs = [
                EGL_WIDTH, 9,
                EGL_HEIGHT, 9,
                EGL_NONE,
            ]
            surface = eglCreatePbufferSurface(eglDpy, config, pbufferAttribs)
            # 4. Bind the API
            eglBindAPI(EGL_OPENGL_API)
            # 5. Create a context and make it current
            eglCtx = eglCreateContext(eglDpy, config, EGL_NO_CONTEXT, None)
            eglMakeCurrent(eglDpy, surface, surface, eglCtx)
        else:
            # create window
            glfw.init()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
            glfw.window_hint(glfw.VISIBLE, False)
            window = glfw.create_window(image_width, image_height, "hidden window", None, None)
            glfw.make_context_current(window)
        
        # vao
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        
        # create the render target
        FramebufferName = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName)
        renderedTexture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, renderedTexture)
        image = np.array([0 for i in range(image_width * image_height * 4)], dtype='ubyte')
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        depthrenderbuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, image_width, image_height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0)
        DrawBuffers = [GL_COLOR_ATTACHMENT0]
        glDrawBuffers(1, DrawBuffers)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print('Failed to create frame buffer!')
            exit(0)
        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName)
        glViewport(0, 0, image_width, image_height)

        # prepare shaders
        ProgramID = load_shaders()
        glUseProgram(ProgramID)

        # set MVP matrix
        # self.Projection = glm.perspective(glm.radians(65), image_width / image_height, 0.01, 100.0)
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], image_height-intrinsics[1, 2]
        far, near = 100.0, 0.01
        left = -cx * (near / fx)
        right = (image_width - cx) * (near / fx)
        bottom = -cy * (near / fy)
        top = (image_height - cy) * (near / fy)
        self.Projection = glm.frustum(left, right, bottom, top, near, far)
        if extrinsics is None:
            extrinsics = np.eye(4)
        self.View = glm.lookAt(
            extrinsics[:3, 3].tolist(),                                                 # eye
            (extrinsics[:3, 3] + extrinsics[:3, :3] @ np.array([0, 0, 1])).tolist(),    # center
            (extrinsics[:3, :3] @ np.array([0, -1, 0])).tolist(),                       # up
        )
        self.MatrixID = dict(
            MVP=glGetUniformLocation(ProgramID, "MVP"), 
            MV=glGetUniformLocation(ProgramID, "MV"), 
            id=glGetUniformLocation(ProgramID, "id"), 
        )
        
        # vertex buffer dict
        self.vertex_buffers = {}
        
        # add table plane
        table_mesh = tm.primitives.Box(extents=[5, 5, 0])
        self.add_object_mesh('table', table_mesh, 0)

    def add_object_mesh(self, object_code, object_mesh, id):
        # prepare data
        vertices = object_mesh.vertices[object_mesh.faces].ravel().tolist()
        vertexbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer)
        glBufferData(GL_ARRAY_BUFFER, (ctypes.c_float*len(vertices))(*vertices), GL_STATIC_DRAW)
        self.vertex_buffers[object_code] = dict(handle=vertexbuffer, len=len(vertices) // 3, id=id)
    
    def render_point_cloud(self, object_code_list, transform_list):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glEnableVertexAttribArray(0)
        for i in range(len(object_code_list)):
            # set MVP matrix
            Model = glm.mat4x4(transform_list[i])
            mvp = self.Projection * self.View * Model
            glUniformMatrix4fv(self.MatrixID['MVP'], 1, GL_FALSE, mvp.to_list())
            glUniformMatrix4fv(self.MatrixID['MV'], 1, GL_FALSE, (self.View * Model).to_list())
            glUniform1i(self.MatrixID['id'], self.vertex_buffers[object_code_list[i]]['id'])
            # Render to framebuffer
            glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffers[object_code_list[i]]['handle'])
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_buffers[object_code_list[i]]['len'])
        glDisableVertexAttribArray(0)
        # get point cloud
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        image = glReadPixels(0, 0, self.image_width, self.image_height, GL_RGBA, GL_FLOAT).reshape(self.image_height, self.image_width, 4)
        glDisableVertexAttribArray(0)
        return image

    # def observe(self, state_dict):
    #     # set MVP matrix
    #     Model = glm.mat4x4(transforms3d.affines.compose(
    #         T=state_dict['object_translation'], 
    #         R=state_dict['object_rotation'], 
    #         Z=[1, 1, 1], 
    #     ))
    #     mvp = self.Projection * self.View * Model
    #     glUniformMatrix4fv(self.MatrixID, 1, GL_FALSE, mvp.to_list())
    #     # Render to framebuffer
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     glEnable(GL_DEPTH_TEST)
    #     glEnableVertexAttribArray(0)
    #     glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffers[state_dict['object_code']]['handle'])
    #     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    #     glDrawArrays(GL_TRIANGLES, 0, self.vertex_buffers[state_dict['object_code']]['len'])
    #     glPixelStorei(GL_PACK_ALIGNMENT, 1)
    #     data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    #     glDisableVertexAttribArray(0)
    #     pc = data.reshape(-1, 4)
    #     pc = pc[pc[:, 3] > 0] # [:, :3]
    #     # result
    #     observation_dict = dict(
    #         pc=torch.tensor(pc, dtype=torch.float), 
    #     )
    #     return observation_dict
