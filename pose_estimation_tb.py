from utils import *
from model import Model
from matplotlib.patches import Circle, Wedge, Polygon
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    nb_img = 5
    path = '..\\..\\Dataset_Generation\\Blender\\result_virtual_dataset_256\\'
    test_file_name = '..\\..\\Dataset_Generation\\Blender\\txt_files\\valid_virtual_256.txt'
    render_res_x = 256
    render_res_y = 256


    HM = np.zeros([64, 64, 3])
    for i in range(nb_img):

        print('------------------------------BEGIN------------------------------')
        #-----------------------------------READING THE TXT FILE------------------------------------
        # camera parameters
        print('--- READING TEXT FILE line :', i, '---')
        name_img, joints, bb, K, RT, kps_3d = read_text_file(test_file_name, i+1)
        P = np.dot(K, RT)
        bb[:, 1] = 256 - bb[:, 1]
        box = point2vertices(bb)
        # P      : the camera parameters matrix, P = K*[R|T]
        # joints :  the keypoint matrix
        # bb     : the bounding box vertices matrix
        # K      : intrinsec parameters of the camera
        # RT     : extrasec parameters of the camera
        # kps_3d : ground truth of the 3d postion of the vertices of the bounding box
        #-------------------------------------------------------------------------------------------

        #------------------------------------READING THE IMAGE--------------------------------------
        print('--- READING IMAGE :',name_img,'---')
        img = cv2.imread(os.path.join(path, name_img))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        #-------------------------------------------------------------------------------------------

        #-----------------------------------READING THE HEATMAPS------------------------------------
        # reading the heatmap
        print('--- READING HEATMAPS ---')
        heatmap = readHM(filepath=path, name=name_img)
        [W, D] = findWMax(heatmap) # /!\ W contains the keypoints of a 64*64 image ! if you want the keypoints for a 256*256 image you have to multiply by 4
        W = 4 * W # the keypoints for a 256*256 image
        W_box = point2vertices(W.T)
        # generate the heatmap image
        response = np.sum(heatmap, 2)
        max_value = np.amax(response)
        min_value = np.amin(response)
        response = (response - min_value) / (max_value - min_value)
        cmap = plt.get_cmap('viridis')
        mapIm = np.delete(cv2.resize(cmap(response), (256, 256)), 3, 2)
        imgtoshow = 0.5 * (img + mapIm * 255) / 255
        #-------------------------------------------------------------------------------------------

        #-------------------------------------LOADING THE MODEL-------------------------------------
        print('--- LOADING MODEL ---')
        model = Model()
        model.load_model()
        #-------------------------------------------------------------------------------------------

        ##--------------------------------------OPTIMIZATION WP--------------------------------------
        #print('--- OPTIMIZATION PROCESS ---')
        #S = model.kp_bb
        #s_gt,s0_gt, R_gt, T_gt = weak_model_opt(np.transpose(bb).astype(np.float64),D,model,verb=0)
        #shape_wp_gt = shape_projection_wp(s_gt, R_gt, T_gt, model, box=True)
        #
        #s_net, s0_net, R_net, T_net = weak_model_opt(W, D, model, verb=0)
        #shape_wp_net = shape_projection_wp(s_net, R_net, T_net, model, box=True)
        ##-------------------------------------------------------------------------------------------
        #
        ##-----------------------------------------DISPLAY-------------------------------------------
        #print('--- RESULT GROUND TRUTH WP ---')
        #print('s : ',s_gt)
        #print('R : \n',R_gt)
        #print('T : \n', T_gt)
        #print('R angle :', anglerotation(R_gt))
        #
        #print()
        #
        #print('--- RESULT NETWORK KEYPOINTS WP ---')
        #print('s : ', s_net)
        #print('R : \n', R_net)
        #print('T : \n', T_net)
        #print('R angle :', anglerotation(R_net))
        ##-------------------------------------------------------------------------------------------
        #
        ##-----------------------------------ERROR COMPUTATION WP------------------------------------
        #print()
        #print('--- ERROR COMPUTATION WP ---')
        #print('relation scale error in %      :', abs((s_net-s_gt)/s_gt)*100)
        #print('translation error in pixel     :', np.linalg.norm(T_net - T_gt, ord=2))
        #print('absolute angle error in degree :', 180 - anglerotation(np.dot(R_gt.T,R_net)))
        #
        ## 3D comparaison of the two rotation matrix
        ##visualrotation(R_gt,R_net)
        ##-------------------------------------------------------------------------------------------

        ##--------------------------------------OPTIMIZATION FP---------------------------------------
        #print('--- OPTIMIZATION PROCESS ---')
        #S = model.kp_bb
        #W_homo = coord2homogenouscoord(W.T).T
        #s, Z, R, T = full_model_opt(W_homo, D, model, verb=0)
        ##--------------------------------------------------------------------------------------------
        #
        ##-----------------------------------------DISPLAY-------------------------------------------
        #print('--- RESULT GROUND TRUTH FP ---')
        #print('s : ',s)
        #print('Z : \n',Z)
        #print('R : \n',R)
        #print('T : \n', T)
        #print('R angle :', anglerotation(R))
        #
        ##-------------------------------------------------------------------------------------------

        #------------------------------------OpenCV OPTIMINIZATION-----------------------------------
        print('--- OpenCV OPTIMINIZATION ---')

        objectPoints = model.kp_bb
        imagePoints_gt = np.transpose(bb).astype(np.float64).T
        imagePoints_net = W.T
        cameraMatrix = K
        distCoeffs = np.zeros((4))

        # pose estimation network kp
        retval_net, rvec_net, tvec_net = cv2.solvePnP(objectPoints, imagePoints_net, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # pose estimation gt
        retval_gt, rvec_gt, tvec_gt = cv2.solvePnP(objectPoints, imagePoints_gt, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # projections network kp
        imagePoints_proj_net, jacobian = cv2.projectPoints(objectPoints, rvec_net, tvec_net, cameraMatrix, distCoeffs)

        # projections gt
        imagePoints_proj_gt, jacobian = cv2.projectPoints(objectPoints, rvec_gt, tvec_gt, cameraMatrix, distCoeffs)


        #-------------------------------------------------------------------------------------------

        ##--------------------------------------------PLOT-------------------------------------------
        print('--- DISPLAYING RESULTS ---')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # image plot and title
        ax1.set_title('Heatmap')
        ax1.imshow(imgtoshow)
        ax2.set_title('Using ground truth keypoints')
        ax2.imshow(img)
        ax3.set_title('Using network keypoints')
        ax3.imshow(img)

        # ax 2
        ax2.plot(imagePoints_proj_gt[:,0,0],imagePoints_proj_gt[:,0,1],'bx')
        ax2.plot(imagePoints_gt[:,0],imagePoints_gt[:,1],'rx')

        polygon21 = Polygon(point2vertices(imagePoints_proj_gt[:,0,:]), linewidth=2, edgecolor='b', facecolor='none')
        ax2.add_patch(polygon21)
        polygon22 = Polygon(point2vertices(imagePoints_gt[:, :]), linewidth=2, edgecolor='r', facecolor='none')
        ax2.add_patch(polygon22)

        # ax 3
        ax3.plot(imagePoints_proj_net[:, 0, 0], imagePoints_proj_net[:, 0, 1], 'bx')
        ax3.plot(imagePoints_net[:, 0], imagePoints_net[:, 1], 'rx')

        polygon31 = Polygon(point2vertices(imagePoints_proj_net[:, 0, :]), linewidth=2, edgecolor='b', facecolor='none')
        ax3.add_patch(polygon31)
        polygon32 = Polygon(point2vertices(imagePoints_net[:, :]), linewidth=2, edgecolor='r', facecolor='none')
        ax3.add_patch(polygon32)

        plt.show()
        #-------------------------------------------------------------------------------------------
        print('-------------------------------END-------------------------------')
