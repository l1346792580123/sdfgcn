# adapted from frankmocap https://github.com/facebookresearch/frankmocap

import numpy as np
import os
from OpenGL.GL import *

# vertices: frames x meshVerNum x 3
# trifaces: facePolygonNum x 3 = 22800 x 3
def ComputeNormal(vertices, trifaces, use_facenormals=False):

    if vertices.shape[0] > 5000:
        print('ComputeNormal: Warning: too big to compute {0}'.format(vertices.shape) )
        return

    #compute vertex Normals for all frames
    U = vertices[:,trifaces[:,1],:] - vertices[:,trifaces[:,0],:]  #frames x faceNum x 3
    V = vertices[:,trifaces[:,2],:] - vertices[:,trifaces[:,1],:]  #frames x faceNum x 3
    originalShape = U.shape  #remember: frames x faceNum x 3

    U = np.reshape(U, [-1,3])
    V = np.reshape(V, [-1,3])
    faceNormals = np.cross(U,V)     #frames x 13776 x 3
    from sklearn.preprocessing import normalize

    if np.isnan(np.max(faceNormals)):
        print('ComputeNormal: Warning nan is detected {0}')
        return
    faceNormals = normalize(faceNormals)

    faceNormals = np.reshape(faceNormals, originalShape)
    if use_facenormals:
        idx = np.zeros(vertices.shape[1],dtype=np.int64)
        for i in range(len(trifaces)):
            idx[trifaces[i,0]] = i
            idx[trifaces[i,1]] = i
            idx[trifaces[i,2]] = i

        vertex_normals = faceNormals[:, idx]
        return vertex_normals

    if False:        #Slow version
        vertex_normals = np.zeros(vertices.shape) #(frames x 11510) x 3
        for fIdx, vIdx in enumerate(trifaces[:,0]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
        for fIdx, vIdx in enumerate(trifaces[:,1]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
        for fIdx, vIdx in enumerate(trifaces[:,2]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
    else:   #Faster version
        # Computing vertex normals, much faster (and obscure) replacement
        index = np.vstack((np.ravel(trifaces), np.repeat(np.arange(len(trifaces)), 3))).T
        index_sorted = index[index[:,0].argsort()]
        vertex_normals = np.add.reduceat(faceNormals[:,index_sorted[:, 1],:][0],
            np.concatenate(([0], np.cumsum(np.unique(index_sorted[:, 0],
            return_counts=True)[1])[:-1])))[None, :]
        vertex_normals = vertex_normals.astype(np.float64)

    originalShape = vertex_normals.shape
    vertex_normals = np.reshape(vertex_normals, [-1,3])
    vertex_normals = normalize(vertex_normals)
    vertex_normals = np.reshape(vertex_normals,originalShape)

    return vertex_normals

# Helper function to locate and open the target file (passed in as a string).
# Returns the full path to the file as a string.
def findFileOrThrow(strBasename):
    # Keep constant names in C-style convention, for readability
    # when comparing to C(/C++) code.
    if os.path.isfile(strBasename):
        return strBasename

    LOCAL_FILE_DIR = "data" + os.sep
    GLOBAL_FILE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep + "data" + os.sep

    strFilename = LOCAL_FILE_DIR + strBasename
    if os.path.isfile(strFilename):
        return strFilename

    strFilename = GLOBAL_FILE_DIR + strBasename
    if os.path.isfile(strFilename):
        return strFilename

    raise IOError('Could not find target file ' + strBasename)

def loadShader(shaderType, shaderFile):
    # check if file exists, get full path name
    strFilename = findFileOrThrow(shaderFile)
    shaderData = None
    with open(strFilename, 'r') as f:
        shaderData = f.read()

    shader = glCreateShader(shaderType)
    glShaderSource(shader, shaderData)  # note that this is a simpler function call than in C

    # This shader compilation is more explicit than the one used in
    # framework.cpp, which relies on a glutil wrapper function.
    # This is made explicit here mainly to decrease dependence on pyOpenGL
    # utilities and wrappers, which docs caution may change in future versions.
    glCompileShader(shader)

    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status == GL_FALSE:
        # Note that getting the error log is much simpler in Python than in C/C++
        # and does not require explicit handling of the string buffer
        strInfoLog = glGetShaderInfoLog(shader)
        strShaderType = ""
        if shaderType is GL_VERTEX_SHADER:
            strShaderType = "vertex"
        elif shaderType is GL_GEOMETRY_SHADER:
            strShaderType = "geometry"
        elif shaderType is GL_FRAGMENT_SHADER:
            strShaderType = "fragment"

        print("Compilation failure for " + strShaderType + " shader:\n" + str(strInfoLog))

    return shader


# Function that accepts a list of shaders, compiles them, and returns a handle to the compiled program
def createProgram(shaderList):
    program = glCreateProgram()

    for shader in shaderList:
        glAttachShader(program, shader)

    glLinkProgram(program)

    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status == GL_FALSE:
        # Note that getting the error log is much simpler in Python than in C/C++
        # and does not require explicit handling of the string buffer
        strInfoLog = glGetProgramInfoLog(program)
        print("Linker failure: \n" + str(strInfoLog))

    for shader in shaderList:
        glDetachShader(program, shader)

    return program