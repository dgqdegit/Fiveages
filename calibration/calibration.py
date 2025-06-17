import os  
import sys  
import math  
import cv2  
import numpy as np  
import trimesh  
import logging  
from arm import Arm  # 请确保更新为你自己的机械臂实现  
from camera import Camera  # 请确保更新为你自己的相机实现 

currentdir = os.path.dirname(os.path.realpath(__file__))  
rootdir = os.path.dirname(os.path.dirname(currentdir))  
sys.path.insert(0, rootdir)  

# 设置日志记录器  
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
logger = logging.getLogger("CalibrationLogger")  

class Calibrator():  
    def __init__(  
            self,  
            calib_L: int,  
            calib_W: int,  
            calib_GRID: float,  
            arm: Arm,  # 使用自定义机械臂类型  
            camera: Camera,  # 使用自定义相机类型   
            save_dir: str,  # 保存路径  
            distCoeffs: np.ndarray  # 相机畸变系数  
    ):  
        """  
        Base calibrator class.  
        """  
        self.L: int = calib_L  
        self.W: int = calib_W  
        self.GRID: float = calib_GRID  
        self.arm: Arm = arm  
        self.camera: Camera = camera   
        self.distCoeffs: np.ndarray = distCoeffs  # 记录畸变系数

        # 创建保存路径
        self.base_save_dir = save_dir  
        self.image_save_dir = os.path.join(self.base_save_dir, "images")  
        self.external_matrix_save_dir = os.path.join(self.base_save_dir, "external_matrices")  

        # 图像子文件夹
        self.rgb_save_dir = os.path.join(self.image_save_dir, "rgb_images")  
        self.corners_save_dir = os.path.join(self.image_save_dir, "corners_images")  

        # 创建目录以保存图像和外参矩阵
        os.makedirs(self.rgb_save_dir, exist_ok=True)  
        os.makedirs(self.corners_save_dir, exist_ok=True)  
        os.makedirs(self.external_matrix_save_dir, exist_ok=True)  

        # 用于存储物体点和图像点
        self.objpoints = []  
        self.imgpoints = []  

    def calib(self, verbose: bool = True, save: bool = False, e2h: bool = True) -> np.ndarray:  
        """  
        Hand Eye calibration. Currently support Eye-to-Hand calibration.  

        Args:  
            verbose: print calibration info verbosely.  
            save: save calibrated camera to base matrix as `calib.npy` in project rootdir.  
            e2h: calibrate with eye-to-hand or eye-in-hand.  

        Returns:  
            T: calibrated base to camera matrix, [4, 4].  
        """  
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  
        objp = np.zeros((self.W * self.L, 3), np.float32)  
        objp[:, :2] = np.mgrid[0: self.L, 0: self.W].T.reshape(-1, 2) * self.GRID  
        
        rvecs = []  
        tvecs = []  
        arm_transes = []  
        arm_rotmat_invs = []  
        i = 0   
        
        while True:  
            rgb = self.camera.get_image()[0]  # 假设get_image返回RGB图像  
            if rgb is None:  
                logger.warning("无法获取图像，继续下一轮...")  
                continue  # 无法获取图像，继续下一轮  
            # 检查图像形状，如果是 RGBA（4 通道），则转换为 RGB（3 通道） 
            if rgb.shape[2] == 4:  
                rgb = rgb[:, :, :3]  # 仅保留 RGB 通道  

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  
            ret, corners = cv2.findChessboardCorners(gray, (self.L, self.W), None)  
            if ret:  
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  
                cv2.drawChessboardCorners(bgr, (self.L, self.W), corners2, ret)  
                cv2.imshow('img', bgr)  
                key = cv2.waitKey(1)  
                if key == ord("q"):  
                    break  
                elif key == ord(" "):  
                    # 保存RGB图像  
                    rgb_filename = os.path.join(self.rgb_save_dir, f"rgb_image_{i}.png")  
                    cv2.imwrite(rgb_filename, rgb)  
                    logger.info(f"保存RGB图像到: {rgb_filename}")  

                    # 保存检测到的角点图像  
                    corners_filename = os.path.join(self.corners_save_dir, f"corners_image_{i}.png")  
                    cv2.imwrite(corners_filename, bgr)  
                    logger.info(f"保存角点图像到: {corners_filename}")   
                    
                    # Solve PnP to get rotation and translation vectors
                    ret, rvec, tvec = cv2.solvePnP(objp, corners2, self.camera.color_intrinsics_mat, self.distCoeffs)  
                    if ret:  
                        self.objpoints.append(objp)  # 保存物体点
                        self.imgpoints.append(corners2)  # 保存图像点
                        rvecs.append(rvec)  
                        tvecs.append(tvec)  
                        i += 1  
                        if verbose:  
                            logger.info(f"测量 #{i}")  
                            logger.info(f"rvec: {rvec.flatten().tolist()}")  
                            logger.info(f"tvec: {tvec.flatten().tolist()}")  
                    else:  
                        if verbose:  
                            logger.warning("无法求解PnP，跳过当前帧。")   
                        continue  

                    # 获取机械臂当前位姿  
                    arm_pose = self.arm.get_pose()[1]  # Meter, radians, sxyz required 
                    arm_pose_list = arm_pose.split(',')  
                    arm_pose = list(map(float, arm_pose_list))  
                    x_m = arm_pose[0] / 1000.0  
                    y_m = arm_pose[1] / 1000.0  
                    z_m = arm_pose[2] / 1000.0  
                    
                    roll = arm_pose[3] * (math.pi / 180)  
                    pitch = arm_pose[4] * (math.pi / 180)  
                    yaw = arm_pose[5] * (math.pi / 180)  
                    
                    arm_pose = [x_m, y_m, z_m, roll, pitch, yaw]  
                    if verbose:  
                        logger.info(f"机械臂位姿: {arm_pose}")  
                    arm_rotmat_inv = trimesh.transformations.euler_matrix(  
                        arm_pose[3], arm_pose[4], arm_pose[5]
                    )[:3, :3].T  
                    
                    arm_transes.append(-arm_rotmat_inv @ arm_pose[:3])  
                    arm_rotmat_invs.append(arm_rotmat_inv)  

        if len(rvecs) < 3:  # 至少需要3个测量  
            logger.error("需要至少3个测量值来完成标定！")  
            return None  

        # 进行Eye-to-Hand 标定
        R, t = cv2.calibrateHandEye(arm_rotmat_invs, arm_transes, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_PARK)  
        T = np.eye(4)  
        T[:3, :3] = R  
        T[:3, 3] = t[:, 0]  

        if verbose:  
            logger.info(f"校正变换矩阵:\n{T}")  
        
        if save:  
            np.save(os.path.join(self.external_matrix_save_dir, "extrinsic_matrix.npy"), T)  
            np.savetxt(os.path.join(self.external_matrix_save_dir, "extrinsic_matrix.txt"), T)  
        return T  

    

if __name__ == "__main__":  
    # 用户输入保存路径和内参
    base_save_dir = os.path.join(os.getcwd(), "calibration_results_right")  # 修改这个保存路径
    calib_L = 5   # 棋盘格内角点数量
    calib_W = 6  
    calib_GRID = 30 / 1000  # 棋盘格大小

    # 相机内参畸变系数
    k1=-0.0486483
    k2=0.0182936
    p1=0.000130559
    p2=-0.000125052
    k3=-0.00746449
    distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)  # 畸变系数

    # 模拟相机和机械臂的初始化  
    camera = Camera()  # 请确保已有相机类  
    camera.start()  # 启动相机  

    arm = Arm(ip="192.168.5.2", port=29999)  # 请修改为你的机械臂IP  
    try:  
        arm.connect()  # 确保连接到机械臂   
        logger.info("已成功连接到机械臂。")  
    except Exception as e:  
        logger.error(f"机械臂连接失败: {str(e)}")  
        exit(1)  # 如果连接失败，则退出程序  

    calibrator = Calibrator(  
        calib_L=calib_L,  
        calib_W=calib_W,  
        calib_GRID=calib_GRID,  
        arm=arm,  
        camera=camera,  
        save_dir=base_save_dir,  # 将保存路径传递给Calibrator 
        distCoeffs=distCoeffs  # 将内参畸变系数传递给Calibrator
    )  
    calibrator.calib(save=True)
