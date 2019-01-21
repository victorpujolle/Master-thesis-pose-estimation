from utils import *
from model import Model
from matplotlib.patches import Circle, Wedge, Polygon
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    nb_img = 1
    path = '..\\..\\Dataset_Generation\\Blender\\result_virtual_dataset_256\\'
    test_file_name = '..\\..\\Dataset_Generation\\Blender\\txt_files\\valid_virtual_256.txt'
    radix_name_img = 'blender_image_num_'
    extension = '.png'
    render_res_x = 256
    render_res_y = 256


    HM = np.zeros([64, 64, 3])
    for i in range(nb_img):

        # name of the image
        img_name = radix_name_img + str(i + 1)
        img = cv2.imread(os.path.join(path, img_name) + extension)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape

        # camera parameters
        name_img, joints, bb, K, RT, kps_3d = read_text_file(test_file_name, i + 1)
        P = np.dot(K, RT)

        # reading the heatmap
        heatmap = readHM(filepath=path, name=name_img)
        [W, D] = findWMax(heatmap) # /!\ W contains the keypoints of a 64*64 image ! if you want the keypoints for a 256*256 image you have to multiply by 4
        W = 4 * W



        print(RT)

        #----------------------------------Weak perspective------------------------------------
        # shape of the model
        model = Model()
        model.load_model()

        # optimization process
        opt_wp = PoseFromKpts_WP2(W, model, D, False)
        opt_wp_gt = PoseFromKpts_WP2(np.transpose(bb).astype(np.float64), model, D, True)

        # weak perspective
        S_wp = np.dot(opt_wp.R, opt_wp.S) # S_wp = sR.S
        print('R wp : \n',opt_wp.R)
        print('S wp : \n',opt_wp.S)
        S_wp[0] += opt_wp.T[0]
        S_wp[1] += opt_wp.T[1]

        # weak perspective
        S_wp_gt = np.dot(opt_wp_gt.R, opt_wp_gt.S)
        S_wp_gt[0] += opt_wp_gt.T[0]
        S_wp_gt[1] += opt_wp_gt.T[1]

        # computation of the polygon weak perspective
        [model_wp, w_wp, R_wp, T_wp] = fullShape(S_wp, model)
        mesh2d_wp = model_wp.vertices[:, 0:2]

        # computation of the polygon weak perspective
        [model_wp_gt, w_wp_gt, R_wp_gt, T_wp_gt] = fullShape(S_wp_gt, model)
        mesh2d_wp_gt = model_wp_gt.vertices[:, 0:2]


        test = w_wp * np.dot(R_wp,np.transpose(model.kp_bb))
        test[0] += T_wp[0]
        test[1] += T_wp[1]
        test[2] += T_wp[2]
        print(np.transpose(test))

        print('R wp : \n', R_wp)
        print('T wp : \n', T_wp)
        print('w wp : \n', w_wp)
        print('W_hp : \n', W)

        print(model.vertices)
        print(model_wp.vertices)

        fig, ax = plt.subplots()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(img)
        polygon = Polygon(mesh2d_wp, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(polygon)
        polygon = Polygon(mesh2d_wp_gt, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)
        plt.show()

