import numpy as np
import cv2
import threading
import time
import open3d as o3d
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R

def get_cam_extrinsic(type):
    if type == "3rd":
        transform =  [[-0.04952737 , 0.41269724 ,-0.90952077 , 0.69906123],
 [ 0.99820036 ,-0.01037279 ,-0.05906304 ,-0.56872775],
 [-0.03380942 ,-0.9108092 , -0.4114408  , 0.56231206],
 [ 0.        ,  0.  ,        0.   ,       1.        ]]
        
    elif type == "wrist":
        trans = np.array([0.6871684912377796, -0.7279882263970943, 0.8123566411202088])
        quat = np.array([-0.869967706085017, -0.2561670369853595, 0.13940123346877276, 0.39762034107764127])
        # Generate transformation matrix from quaternion
        rot = R.from_quat(quat)
        rot_matrix = rot.as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = trans
    else:
        raise ValueError("Invalid type")
    
    return transform
 
class ZedCam:
    def __init__(self, serial_number, resolution=None):
        self.zed = sl.Camera()
        self.init_zed(serial_number=serial_number)
        
        if resolution:
            self.img_size = sl.Resolution()
            self.img_size.height = resolution[0]
            self.img_size.width = resolution[1]
        else:
            self.img_size = self.zed.get_camera_information().camera_configuration.resolution
            
        self.center_crop = False
        self.center_crop_size = (480, 640)

    def init_zed(self, serial_number):
        # Initialization parameters
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : " + repr(err) + ". Exit program.")
            exit()
            
        # Init 50 frames
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        for _ in range(50):
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                
    def capture(self):
        image = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)
        point_cloud = sl.Mat()

        while True:
            runtime_parameters = sl.RuntimeParameters()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, self.img_size)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.img_size)
                frame_timestamp_ms = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_microseconds()
                break
            
        rgb_image = image.get_data()[..., :3]
        pcd = point_cloud.get_data()
        pcd[np.isnan(pcd)] = 0
        pcd = pcd[..., :3] * 0.001  # Convert to meters
        
        result_dict = {
            "rgb": rgb_image,
            "pcd": pcd,
            "timestamp_ms": frame_timestamp_ms / 1000.0,
        }
        return result_dict
    
    def stop(self):
        self.zed.close()
        
class Camera:
    def __init__(self, camera_type="all", timestamp_tolerance_ms=80):
        static_serial_number = 32293157   # 更新为您自己的相机序列号
        wrist_serial_number = 31660984

        if camera_type == "all":
            self.cams = [ZedCam(serial_number=static_serial_number), ZedCam(serial_number=wrist_serial_number)]
            self.camera_types = ["3rd", "wrist"]
        elif camera_type == "3rd":
            self.cams = [ZedCam(serial_number=static_serial_number)]
            self.camera_types = ["3rd"]
        elif camera_type == "wrist":
            self.cams = [ZedCam(serial_number=wrist_serial_number)]
            self.camera_types = ["wrist"]
        else:
            raise ValueError("Invalid camera type, please choose from 'all', '3rd', 'wrist'")
        
        self.timestamp_tolerance_ms = timestamp_tolerance_ms
        
    def _capture_frame(self, idx, result_dict, start_barrier, done_barrier):
        cam = self.cams[idx]
        camera_type = self.camera_types[idx]
        start_barrier.wait()  # 同步
        result_dict[camera_type] = cam.capture()
        done_barrier.wait()
        
    def capture_frames_multi_thread(self):
        result_dict = {}
        num_cameras = len(self.cams)

        if num_cameras == 1:
            result_dict[self.camera_types[0]] = self.cams[0].capture()
            _ = [result_dict[cam].pop("timestamp_ms", None) for cam in result_dict]
            return result_dict
        
        else:
            start_barrier = threading.Barrier(num_cameras)
            done_barrier = threading.Barrier(num_cameras)
            threads = []

            for idx in range(num_cameras):
                t = threading.Thread(
                    target=self._capture_frame,
                    args=(idx, result_dict, start_barrier, done_barrier)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Timestamp alignment step
            timestamps = [result_dict[cam]["timestamp_ms"] for cam in result_dict]
            _ = [result_dict[cam].pop("timestamp_ms", None) for cam in result_dict]
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            diff_ts = max_ts - min_ts

            if diff_ts > self.timestamp_tolerance_ms:
                print("Timestamps are not aligned, difference is", diff_ts, "ms, discard frames")
                return None
            else:
                return result_dict
    
    def capture(self):
        while True:
            result_dict = self.capture_frames_multi_thread()
            if result_dict is not None:
                break
        return result_dict
    
    def stop(self):
        for cam in self.cams:
            cam.stop()

if __name__ == "__main__":
    cameras = Camera(camera_type="3rd")
    time.sleep(2)

    observation = cameras.capture()
    observation["3rd"]["rgb"] = observation["3rd"]["rgb"][:, :, ::-1].copy()  # BGR to RGB

    def convert_pcd_to_base(type="3rd", pcd=[]):
        transform = get_cam_extrinsic(type)
        
        h, w = pcd.shape[:2]
        pcd = pcd.reshape(-1, 3)
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)  # 同齐
        pcd = (transform @ pcd.T).T[:, :3]
        
        pcd = pcd.reshape(h, w, 3)
        return pcd 
    
    def vis_pcd(pcd, rgb):
        # Flatten point cloud and RGB image
        pcd_flat = pcd.reshape(-1, 3)  # (N, 3)
        rgb_flat = rgb.reshape(-1, 3) / 255.0  # Normalize RGB values
        
        # Create Open3D PointCloud
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pcd_flat)
        pcd_obj.colors = o3d.utility.Vector3dVector(rgb_flat)

        # Display origin coordinate frame
        axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd_obj, axis_origin])

        # Save to PLY file
        o3d.io.write_point_cloud("output.ply", pcd_obj)

    observation["3rd"]["pcd"] = convert_pcd_to_base("3rd", observation["3rd"]["pcd"])
    vis_pcd(observation["3rd"]["pcd"], observation["3rd"]["rgb"])

    print("Point cloud saved to output.ply")