import numpy as np

class Model:
    """
    the model of the bounding box is implemented here
    """

    def __init__(self):

        self.nb_faces = 0
        self.nb_vertices = 0
        self.faces = []
        self.vertices = []
        self.kp_name = ["frd","fru","flu","fld","brd","bru","blu","bld"] # name of the joints
        self.scale = np.array([0.112, 0.1, 0.074]) # the scale of the box
        self.kp_bb = np.array([
            [ self.scale[0] / 2, -self.scale[1] / 2,  self.scale[2] / 2],
            [ self.scale[0] / 2,  self.scale[1] / 2,  self.scale[2] / 2],
            [-self.scale[0] / 2,  self.scale[1] / 2,  self.scale[2] / 2],
            [-self.scale[0] / 2, -self.scale[1] / 2,  self.scale[2] / 2],
            [ self.scale[0] / 2, -self.scale[1] / 2, -self.scale[2] / 2],
            [ self.scale[0] / 2,  self.scale[1] / 2, -self.scale[2] / 2],
            [-self.scale[0] / 2,  self.scale[1] / 2, -self.scale[2] / 2],
            [-self.scale[0] / 2, -self.scale[1] / 2, -self.scale[2] / 2]])

        self.kp_mug = 0.1*np.array(
            [[0, 3.6, 1.6, 0, 0, 0],
            [0, 7.1, 5.2, 0, 0, 0],
            [0, 3.6, 8.8, 0, 0, 0],
            [0, 3.3, 10.2, 0, 0, 0],
            [-3.3, 0, 10.2, 0, 0, 0],
            [0, -3.3, 10.2, 0, 0, 0],
            [3.3, 0, 10.2, 0, 0, 0],
            [0, 0, 0.2, 0, 0, 0]]
        )

        self.S0 = np.transpose(self.kp_bb); # shape matrix

        S = np.transpose(self.kp_bb);

        self.mu = np.copy(S) # unnormalizaed vesrion of S

        mean = np.mean(S, 1)
        for i in range(3):
            S[i] -= mean[i]
        std = np.std(S,1)
        a = np.mean(std)
        S = S / a
        self.B = S # normalized shape matrix

    def load_model(self, path='data_files', name_file_faces='cad_faces.csv', name_file_vertices='cad_vertices2.csv'):
        # loads the cad model into self.faces and self.vertices

        # faces loading
        self.faces = []
        name_faces = path + '/' + name_file_faces

        file_faces = open(name_faces, 'r')
        count = 0

        for lines in file_faces:
                line = lines.split(',')
                self.faces.append([])

                for x in line:
                    self.faces[count].append(float(x))

                count += 1

        file_faces.close()

        # vertices loading
        self.vertices = []
        name_vertices = path + '/' + name_file_vertices

        files_vertices = open(name_vertices, 'r')
        count = 0

        for lines in files_vertices:
                line = lines.split(',')
                self.vertices.append([])

                for x in line:
                    self.vertices[count].append(float(x))

                count += 1

        files_vertices.close()

        self.nb_faces = len(self.faces)
        self.nb_vertices = len(self.vertices)

        self.vertices = np.array(self.vertices)
        self.faces = np.array(self.faces)

    def copy(self):
        model_copy = Model()
        model_copy.nb_faces = self.nb_faces
        model_copy.nb_vertices = self.nb_vertices
        model_copy.faces = np.copy(self.faces)
        model_copy.vertices = np.copy(self.vertices)
        model_copy.scale = np.copy(self.scale)
        model_copy.kp_bb = np.copy(self.kp_bb)

        return model_copy

class Output:
    '''
    An object of this class represents the output of the function PoseFromKpts_WP or PoseFromKpts_FP
    '''

    def __init__(self, S=[], M=[], R=[], C=[], C0=[], T=[], Z=[], fval=0):
        self.S = S
        self.M = M
        self.R = R
        self.C = C
        self.C0 = C0
        self.T = T
        self.Z = Z
        self.fval = fval

class Store:
    '''
    usefull class to pass modifiable local parameters from function to function 
    '''
    def __init__(self):
        self.stored = None
