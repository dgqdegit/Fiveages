import socket  
import struct  
import threading  
import time  
from queue import Queue, Full  
import argparse  
import binascii  
import os  
from datetime import datetime  
import cv2  
import pyrealsense2 as rs  
import numpy as np  
import pyzed.sl as sl  
from pymodbus.client import ModbusTcpClient  
from pynput import keyboard  
import pickle  
from PIL import Image  
import shutil 



class ZedCam:  
    def __init__(self, serial_number, resolution=None):  
        self.zed = sl.Camera()  
        self.init_zed(serial_number)  

        if resolution:  
            self.img_size = sl.Resolution()  
            self.img_size.height = resolution[0]  
            self.img_size.width = resolution[1]  
        else:  
            self.img_size = self.zed.get_camera_information().camera_configuration.resolution  

    def init_zed(self, serial_number):  
        init_params = sl.InitParameters()  
        init_params.set_from_serial_number(serial_number)  
        init_params.camera_resolution = sl.RESOLUTION.HD1080  
        init_params.camera_fps = 30  
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL  
        init_params.coordinate_units = sl.UNIT.MILLIMETER  

        err = self.zed.open(init_params)  
        if err != sl.ERROR_CODE.SUCCESS:  
            print("Camera Open : " + repr(err) + ". Exit program.")  
            exit()  

        # Init 50 frames  
        image = sl.Mat()  
        for _ in range(50):  
            runtime_parameters = sl.RuntimeParameters()  
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  
                self.zed.retrieve_image(image, sl.VIEW.LEFT)  

    def capture(self):  
        image = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)  
        depth_map = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)  
        point_cloud = sl.Mat()  

        while True:  
            runtime_parameters = sl.RuntimeParameters()  
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                print("if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS")
                self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, self.img_size)  
                self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU, self.img_size)  
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.img_size)  
                frame_timestamp_ms = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_microseconds()  
                break  

        rgb_image = image.get_data()[..., :3]
        print("rgb_image = image.get_data()[..., :3]")
        depth = depth_map.get_data()  
        depth[np.isnan(depth)] = 0  
        depth_image_meters = depth * 0.001  
        pcd = point_cloud.get_data()  
        pcd[np.isnan(pcd)] = 0  
        pcd = pcd[..., :3] * 0.001  

        return {  
            "rgb": rgb_image,  
            "depth": depth_image_meters,  
            "pcd": pcd,  
            "timestamp_ms": frame_timestamp_ms / 1000.0,  
        }  

    def stop(self):  
        self.zed.close()  


class CR5Realtime:  
    def __init__(self, ip="192.168.5.100", port=30004, frequency=None, base_save_path=None,  
                 realsense_serial="327122077302", zed_serial="32293157", instruction="put_bottle_in_microwave",  
                 extrinsics_file="/home/zk/Public/extrinsic_matrix.npy"):  
        self.base_save_path = base_save_path or os.path.join(os.getcwd(), 'data', 'right_wbl_1')  
        os.makedirs(self.base_save_path, exist_ok=True)  

        self.save_counter = 0  
        self.ip = ip  
        self.port = port  
        self.running = False  
        self.data_queue = Queue(maxsize=100)  
        self.callbacks = []  
        self.realsense_camera = self._init_realsense_camera(realsense_serial)  
        self.zed_camera = ZedCam(serial_number=zed_serial)  
        self.last_pose = None  
        self.dragging = False  
        self.claw_status = None  
        self.last_claw_status = None  
        self.save_lock = threading.Lock()  
        self.latest_tcp_pose = None  
        self.key_listener = None  
        self.instruction = instruction  
        self.extrinsics_file_path = extrinsics_file

        if frequency is None:  
            if port == 30004:  
                self.frequency = 125  
            elif port == 30005:  
                self.frequency = 5  
            elif port == 30006:  
                self.frequency = 20  
            else:  
                self.frequency = 10  
        else:  
            self.frequency = frequency  

        self.period = 1.0 / self.frequency  
        self._stats = {  
            'packets_received': 0,  
            'packets_processed': 0,  
            'last_latency': 0,  
            'max_latency': 0,  
            'start_time': 0  
        }  

        print(f"Initializing CR5 connection, IP: {ip}, Port: {port}, Frequency: {self.frequency}Hz")  
        self._connect()  

        self.folder_number = self._find_next_folder_number()  
        self.folder_path = os.path.join(self.base_save_path, str(self.folder_number))  
        os.makedirs(self.folder_path, exist_ok=True)  

        self.actions_folder = os.path.join(self.folder_path, "actions")  
        os.makedirs(self.actions_folder, exist_ok=True)  
        
        self.instruction_file_path = os.path.join(self.folder_path, "instruction.txt")  

        with open(self.instruction_file_path, 'w') as f:  
            f.write(self.instruction)  

        # 保存指令为PKL格式
        self.save_instruction_as_pkl()

        # 直接复制外参矩阵到保存路径  
        extrinsics_save_path = os.path.join(self.folder_path, "extrinsic_matrix.npy")
        shutil.copy(self.extrinsics_file_path, extrinsics_save_path)  # 复制外参矩阵
        print(f"Extrinsics matrix copied to: {extrinsics_save_path}")

    def _find_next_folder_number(self):
        """Find the next available folder number."""
        existing_folders = [d for d in os.listdir(self.base_save_path) if d.isdigit()]
        return len(existing_folders)  # 统计现有数字文件夹的数量
    
    def save_instruction_as_pkl(self):
        """将 instruction.txt 转换为 PKL 格式。"""
        instruction_data = self.instruction  # 使用存储的指令
        instruction_path = self.instruction_file_path.replace('.txt', '.pkl') 
        with open(instruction_path, 'wb') as pkl_file:
            pickle.dump(instruction_data, pkl_file)  # 将指令内容存储为PKL格式
        print(f"Converted instruction to PKL format and saved to {instruction_path}")  # 打印保存路径


    def _init_realsense_camera(self, serial_number):  
        try:  
            pipeline = rs.pipeline()  
            config = rs.config()  
            config.enable_device(serial_number)  
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  
            profile = pipeline.start(config)  
            depth_sensor = profile.get_device().first_depth_sensor()  
            depth_scale = depth_sensor.get_depth_scale()  
            print(f"RealSense Depth Scale is: {depth_scale}")  
            return {  
                'pipeline': pipeline,  
                'profile': profile,  
                'depth_scale': depth_scale  
            }  
        except Exception as e:  
            print(f"Failed to initialize RealSense camera: {e}")  
            return None  

    def _connect(self):  
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.sock.settimeout(5.0)  
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  
        try:  
            self.sock.connect((self.ip, self.port))  
            print(f"Connected to CR5 robot arm: {self.ip}:{self.port}")  
            self._verify_data_stream()  
        except Exception as e:  
            print(f"Connection failed: {e}")  
            if self.port != 30004:  
                print(f"Try connecting to the standard real-time data port 30004")  
            raise  

    def _verify_data_stream(self):  
        try:  
            self.sock.settimeout(2.0)  
            data = self.sock.recv(1440)  
            if len(data) == 0:  
                print("Warning: Connected but no data received, check robot arm status")  
            else:  
                print(f"Successfully received data: {len(data)} bytes")  
                self._debug_data_format(data)  
        except socket.timeout:  
            print("Warning: No data stream received, robot arm may not be enabled or in error state")  
        except Exception as e:  
            print(f"Error verifying data stream: {e}")  

    def _debug_data_format(self, data):  
        try:  
            msg_size = struct.unpack('<H', data[0:2])[0]  
            print(f"Message length of packet: {msg_size} bytes")  
            di_bytes = binascii.hexlify(data[8:16]).decode()  
            do_bytes = binascii.hexlify(data[16:24]).decode()  
            print(f"Digital input state (Hex): {di_bytes}")  
            print(f"Digital output state (Hex): {do_bytes}")  

            robot_mode = struct.unpack('<Q', data[24:32])[0]  
            print(f"Robot mode: {robot_mode}")  

            joint_positions = struct.unpack('<6d', data[432:480])  
            print("Joint positions (radians):")  
            for i, pos in enumerate(joint_positions):  
                print(f"  Joint {i + 1}: {pos:.4f} rad = {pos:.2f}°")  

            tcp_pose = struct.unpack('<6d', data[624:672])  
            print("TCP position (mm/degrees):")  
            print(f"  X: {tcp_pose[0]:.2f}, Y: {tcp_pose[1]:.2f}, Z: {tcp_pose[2]:.2f}")  
            print(f"  Rx: {tcp_pose[3]:.2f}, Ry: {tcp_pose[4]:.2f}, Rz: {tcp_pose[5]:.2f}")  

            print("Data packet format verification successful, can be parsed normally")  
        except Exception as e:  
            print(f"Failed to analyze packet format: {e}")  

    def pose_callback(self, data):  
        joints = data['joint_actual']  
        j_deg = [j * 57.2958 for j in joints]  

        cart = data['tcp_actual']  
        mode = data['robot_mode']  
        mode_str = ROBOT_MODES.get(mode, f"Unknown ({mode})")  

    def start(self):  
        self.running = True  
        self._stats['start_time'] = time.perf_counter()  

        self.recv_thread = threading.Thread(target=self._recv_loop, name="CR5-Receiver")  
        self.recv_thread.daemon = True  

        self.process_thread = threading.Thread(target=self._process_loop, name="CR5-Processor")  
        self.process_thread.daemon = True  

        self.recv_thread.start()  
        self.process_thread.start()  
        print(f"Real-time data stream started, processing frequency: {self.frequency}Hz")  

    def _recv_loop(self):  
        buffer = b''  
        packet_size = 1440  

        while self.running:  
            try:  
                chunk = self.sock.recv(4096)  
                if not chunk:  
                    print("Connection closed, attempting to reconnect...")  
                    self._reconnect()  
                    continue  

                buffer += chunk  

                while len(buffer) >= packet_size:  
                    packet = buffer[:packet_size]  
                    buffer = buffer[packet_size:]  

                    try:  
                        self.data_queue.put((time.perf_counter(), packet), block=False)  
                        self._stats['packets_received'] += 1  
                    except Full:  
                        try:  
                            self.data_queue.get_nowait()  
                            self.data_queue.put((time.perf_counter(), packet), block=False)  
                        except:  
                            pass  

            except (socket.timeout, ConnectionResetError) as e:  
                print(f"Connection error: {e}")  
                self._reconnect()  

    def _process_loop(self):  
        next_time = time.perf_counter() + self.period  

        while self.running:  
            current_time = time.perf_counter()  

            if current_time >= next_time:  
                try:  
                    timestamp, packet = self.data_queue.get(block=False)  
                    self._process_packet(timestamp, packet)  
                    next_time = current_time + self.period  
                    self._stats['packets_processed'] += 1  
                except Exception as e:  
                    next_time = current_time + self.period  
            else:  
                sleep_time = next_time - current_time  
                if sleep_time > 0.001:  
                    time.sleep(sleep_time * 0.8)  

    def _process_packet(self, timestamp, data):  
        try:  
            tcp_actual = struct.unpack('<6d', data[624:672])  
            robot_mode = struct.unpack('<Q', data[24:32])[0]  
            self.latest_tcp_pose = tcp_actual  
            is_dragging = robot_mode == 7  

            # 更新夹爪状态  
            claw_status_flag, claw_status = self._read_claw_status()  
            self._update_claw_status(claw_status)  

            if is_dragging != self.dragging:  
                if is_dragging:  
                    print("Detected operation, start recording...")  
                    self.dragging = True  
                else:  
                    print("Operation stopped, stop recording...")  
                    self.dragging = False  

            # 调用注册的回调  
            for callback in self.callbacks:  
                callback({  
                    'joint_actual': tcp_actual[:6],  
                    'tcp_actual': tcp_actual,  
                    'robot_mode': robot_mode  
                })  

        except Exception as e:  
            print(f"Error processing data: {e}")  

    def manual_save(self):  
        """Save data and images triggered by key press"""  
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  
        with self.save_lock:  
            if self.latest_tcp_pose is not None:  
                self._save_data_and_image(self.latest_tcp_pose)  
            else:  
                print("No TCP pose available for saving.")  

    def start_keyboard_listener(self):  
        """Start keyboard listener thread"""  
        def on_press(key):  
            try:  
                if key.char == 'a':  
                    print("Key 'a' pressed! Saving data...")  
                    self.manual_save()  
            except AttributeError:  
                pass  # Non-character key  

        self.key_listener = keyboard.Listener(on_press=on_press)  
        self.key_listener.start()  

    def _update_claw_status(self, new_status):  
        """Update claw status and check if it has changed"""  
        self.claw_status = new_status  # Update the status  

    def _read_claw_status(self) -> tuple:  
        """Read the claw status via Modbus."""  
        client = ModbusTcpClient(self.ip, port=502)  
        if not client.connect():  
            return False, "Failed to connect to Modbus server."  

        try:  
            unit_id = 2  
            read_address = 258  
            read_response = client.read_holding_registers(read_address, count=1, unit=unit_id)  

            if not read_response.isError():  
                return True, read_response.registers[0]  
            else:  
                return False, str(read_response)  
        except Exception as err:  
            print(f"Error reading claw status: {err}")  
            return False, err  
        finally:  
            client.close()  

    def euler_xyz_to_quaternion(self, roll, pitch, yaw):  
        """Convert Euler angles (XYZ order) to quaternion."""  
        cx = np.cos(roll / 2)  
        sx = np.sin(roll / 2)  
        cy = np.cos(pitch / 2)  
        sy = np.sin(pitch / 2)  
        cz = np.cos(yaw / 2)  
        sz = np.sin(yaw / 2)  

        # Compute quaternion  
        w = cx * cy * cz - sx * sy * sz  
        x = sx * cy * cz + cx * sy * sz  
        y = cx * sy * cz - sx * cy * sz  
        z = cx * cy * sz + sx * sy * cz  

        return (w, x, y, z)  

    def _save_data_and_image(self, tcp_actual):  
        print("开始保存数据")  
        try:  
            # 解包位置和欧拉角  
            position_x, position_y, position_z, roll, pitch, yaw = tcp_actual  
            print(position_x, position_y, position_z, roll, pitch, yaw)
            claw_status = self.claw_status  # 使用当前的夹爪状态  

            # 将位置从毫米转换为米  
            position_x *= 0.001  
            position_y *= 0.001  
            position_z *= 0.001  

            # 将欧拉角从度转换为弧度  
            roll_rad = np.radians(roll)  
            pitch_rad = np.radians(pitch)  
            yaw_rad = np.radians(yaw)  
            
            # 转换为四元数  
            import transforms3d
            euler_ = [roll_rad, pitch_rad, yaw_rad]
            quaternion = transforms3d.euler.euler2quat(*euler_, axes='sxyz')
            # quaternion = self.euler_xyz_to_quaternion(roll_rad, pitch_rad, yaw_rad)  

            
            # 准备保存的姿态数据  
            pose_data = [position_x, position_y, position_z, *quaternion, claw_status]  
            
            # 根据保存计数生成文件名  
            txt_filename = f"{self.save_counter}.txt"  
            txt_filepath = os.path.join(self.actions_folder, txt_filename)  # 保存到 actions 目录  

            # 将姿态数据保存到单独的 .txt 文件  
            np.savetxt(txt_filepath, [pose_data], fmt='%.6f')  

            # 转换为 .pkl 格式  
            pkl_filepath = txt_filepath.replace('.txt', '.pkl')  
            with open(pkl_filepath, 'wb') as pkl_file:  
                np_array = np.array(pose_data)  
                pickle.dump(np_array, pkl_file)  

            os.remove(txt_filepath)

            # 捕获并保存图像  
            self._capture_and_save_realsense_image()  
            self._capture_and_save_zed_image()  

            # 更新保存计数器  
            self.save_counter += 1  
            print(f"第 {self.save_counter} 条数据已保存完毕。")  

        except Exception as e:  
            print(f"错误保存数据和图像: {e}")

    def _capture_and_save_realsense_image(self):  
        if not self.realsense_camera:  
            print("RealSense camera not initialized, cannot capture image.")  
            return  

        try:  
            frames = self.realsense_camera['pipeline'].wait_for_frames()  
            rgb_frame = frames.get_color_frame()  
            rgb_image = np.asanyarray(rgb_frame.get_data())  

            # Save RGB Image as PNG
            save_path = os.path.join(self.folder_path, "wrist_cam_imgs")  
            os.makedirs(save_path, exist_ok=True)  
            image_count = len(os.listdir(save_path))  
            image_path = os.path.join(save_path, f"{image_count}.png")
            cv2.imwrite(image_path, rgb_image)
            print(f"RealSense RGB image saved as PNG to: {image_path}")

            # Convert RGB Image to PKL
            rgb_pkl_folder = os.path.join(self.folder_path, "wrist_cam_rgb")
            os.makedirs(rgb_pkl_folder, exist_ok=True)
            rgb_pkl_path = os.path.join(rgb_pkl_folder, f"{image_count}.pkl")
            with open(rgb_pkl_path, 'wb') as rgb_pkl_file:
                pickle.dump(rgb_image, rgb_pkl_file)
            print(f"RealSense RGB image converted and saved to PKL: {rgb_pkl_path}")

        except Exception as e:
            print(f"Error capturing and saving RealSense image: {e}")

    def _capture_and_save_zed_image(self):
        if not self.zed_camera:
            print("ZED camera not initialized, cannot capture image.")
            return

        try:
            result = self.zed_camera.capture()
            rgb_image = result["rgb"]   # bgr
            import copy
            rgb_image2= copy.deepcopy(rgb_image)
            # cv2.imshow('First Frame', rgb_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            depth_image = result["depth"]
            pcd_data = result["pcd"]

            # Save RGB Image
            save_path = os.path.join(self.folder_path, "3rd_cam_imgs")
            os.makedirs(save_path, exist_ok=True)
            image_count = len(os.listdir(save_path))
            image_path_rgb = os.path.join(save_path, f"{image_count}.png")

            # Save as PNG using PIL (properly handling the path)
            rgb_image2 = cv2.cvtColor(rgb_image2, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(rgb_image2)
            img.save(image_path_rgb)  # Fixed: using the actual path instead of "img.png"
            print(f"ZED RGB image saved to: {image_path_rgb}")

            # Read back the image to verify it was saved correctly
            rgb_image_from_file = cv2.imread(image_path_rgb)

            # Convert RGB Image to PKL
            rgb_pkl_folder = os.path.join(self.folder_path, "3rd_cam_rgb")
            os.makedirs(rgb_pkl_folder, exist_ok=True)
            rgb_pkl_path = os.path.join(rgb_pkl_folder, f"{image_count}.pkl")

            # Save the original array (rgb_image2) instead of re-loaded version for better consistency
            with open(rgb_pkl_path, 'wb') as rgb_pkl_file:
                # If you need BGR format (OpenCV default), uncomment next line
                # rgb_image2 = cv2.cvtColor(rgb_image2, cv2.COLOR_RGB2BGR)
                pickle.dump(rgb_image2, rgb_pkl_file)  # Using original array instead of re-loaded version
            print(f"Zed RGB image converted and saved to PKL: {rgb_pkl_path}")
            
            # with open(rgb_pkl_path, 'rb') as f:
            #     loaded_img = pickle.load(f)
            # cv2.imshow('loaded Frame', loaded_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Save Depth Image
            save_path_depth = os.path.join(self.folder_path, "3rd_cam_depth")
            os.makedirs(save_path_depth, exist_ok=True)
            depth_image_count = len(os.listdir(save_path_depth))
            depth_npy_path = os.path.join(save_path_depth, f"{depth_image_count}.npy")
            np.save(depth_npy_path, depth_image)
            print(f"ZED depth image saved to: {depth_npy_path}")

            # Convert and Save Depth Image to PKL
            depth_pkl_path = depth_npy_path.replace('.npy', '.pkl')
            with open(depth_pkl_path, 'wb') as pkl_file:
                pickle.dump(depth_image, pkl_file)
            print(f"ZED depth image converted and saved to: {depth_pkl_path}")
            os.remove(depth_npy_path)
            
            # Save PCD Data
            save_path_pcd = os.path.join(self.folder_path, "3rd_cam_pcd")
            os.makedirs(save_path_pcd, exist_ok=True)
            pcd_image_count = len(os.listdir(save_path_pcd))
            pcd_npy_path = os.path.join(save_path_pcd, f"{pcd_image_count}.npy")
            np.save(pcd_npy_path, pcd_data)
            print(f"ZED point cloud data saved to: {pcd_npy_path}")

            # Convert and Save PCD Data to PKL
            pcd_pkl_path = pcd_npy_path.replace('.npy', '.pkl')
            with open(pcd_pkl_path, 'wb') as pkl_file:
                pickle.dump(pcd_data, pkl_file)
            print(f"ZED point cloud data converted and saved to: {pcd_pkl_path}")

            
            os.remove(pcd_npy_path)
        except Exception as e:
            print(f"Error capturing and saving ZED images: {e}")
    
 

    def convert_instruction_to_pkl(self):
        """Convert instruction.txt to PKL format."""
        instruction_data = self.instruction  # Use stored instruction
        pkl_path = self.instruction_file_path.replace('.txt', '.pkl')
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(instruction_data, pkl_file)
        print(f"Converted instruction to PKL format and saved to {pkl_path}")

    def _reconnect(self):
        retry_count = 0
        max_retries = 5

        while self.running and retry_count < max_retries:
            retry_count += 1
            print(f"Attempting to reconnect ({retry_count}/{max_retries})...")
            try:
                self.sock.close()
                time.sleep(1)
                self._connect()
                print("Reconnected successfully!")
                return
            except Exception as e:
                print(f"Reconnect failed: {e}")

        print("Maximum reconnection attempts exceeded, giving up")
        self.running = False

    def register_callback(self, callback):
        self.callbacks.append(callback)
        return len(self.callbacks) - 1

    def get_stats(self):
        run_time = time.perf_counter() - self._stats['start_time']
        actual_freq = self._stats['packets_processed'] / run_time if run_time > 0 else 0

        return {
            'uptime': f"{run_time:.1f} seconds",
            'packets_received': self._stats['packets_received'],
            'packets_processed': self._stats['packets_processed'],
            'set_frequency': f"{self.frequency}Hz",
            'actual_frequency': f"{actual_freq:.1f}Hz",
            'current_latency': f"{self._stats['last_latency']:.2f}ms",
            'max_latency': f"{self._stats['max_latency']:.2f}ms",
            'queue_size': f"{self.data_queue.qsize()}/{self.data_queue.maxsize}"
        }

    def stop(self):
        self.running = False
        self.dragging = False
        self.last_pose = None
        self.sock.close()
        if self.realsense_camera:
            self.realsense_camera['pipeline'].stop()
        if self.zed_camera:
            self.zed_camera.stop()
        print("CR5 data collection has stopped, all cameras have been closed.")

    


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CR5 robotic arm real-time data monitoring and dual camera image acquisition')
    parser.add_argument('--ip', default='192.168.5.2', help='Robot arm IP address')
    parser.add_argument('--port', type=int, default=30004, help='Data port (30004/30005/30006)')
    parser.add_argument('--freq', type=int, default=10, help='Data processing frequency (Hz), default to 10Hz')
    parser.add_argument('--realsense_serial', default='327122077302', help='RealSense serial number')
    parser.add_argument('--zed_serial', default='32293157', help='ZED camera serial number')  # 注意相机序列号
    parser.add_argument('--extrinsics_file', default='/home/zk/Projects/camera_calibration/calibration_results_right/external_matrices/extrinsic_matrix.npy', help='Extrinsics matrix file path (npy)') # 更改外参矩阵路径
    parser.add_argument('--instruction', default='put zebra in the drawer', help='Instruction to execute') # 更改语言指令
    args = parser.parse_args()

    # Create base save path
    # base_save_path = os.path.join(os.getcwd(), 'data', 'right_wbl_0604')
    base_save_path = os.path.join('/home/zk/Projects/Datasets', 'put_drawer_0611_right')  # 更改数据保存路径
    os.makedirs(base_save_path, exist_ok=True)

    ROBOT_MODES = {
        1: "Initialization State (ROBOT_MODE_INIT)",
        2: "Brake Open (ROBOT_MODE_BRAKE_OPEN)",
        3: "Power off (ROBOT_MODE_POWEROFF)",
        4: "Disabled (ROBOT_MODE_DISABLED)",
        5: "Idle (ROBOT_MODE_ENABLE)",
        6: "Backdrive Mode (ROBOT_MODE_BACKDRIVE)",
        7: "Running (ROBOT_MODE_RUNNING)",
        8: "Single Move (ROBOT_MODE_SINGLE_MOVE)",
        9: "Error Status (ROBOT_MODE_ERROR)",
        10: "Paused (ROBOT_MODE_PAUSE)",
        11: "Collision Triggered (ROBOT_MODE_COLLISION)"
    }

    try:
        cr5 = CR5Realtime(
            ip=args.ip,
            port=args.port,
            frequency=args.freq,
            base_save_path=base_save_path,
            realsense_serial=args.realsense_serial,
            zed_serial=int(args.zed_serial),
            instruction=args.instruction,
            extrinsics_file=args.extrinsics_file
            
        )

        cr5.register_callback(cr5.pose_callback)  # Register the pose callback
        cr5.start()
        cr5.start_keyboard_listener()
        print("Press Ctrl+C to stop collection...")

        while True:
            time.sleep(5)
            stats = cr5.get_stats()
            # print(stats)

    except ConnectionRefusedError:
        print("\nConnection refused! Please check the following steps:")
        print("1. Ensure the robot arm is powered and the network cable is connected")
        print("2. Ensure the robot arm is enabled (via the teach pendant)")
        print("3. Try connecting to other ports: 30005 (200ms) or 30006 (50ms)")
        print("4. Check if the computer's IP settings are in the 192.168.5.x range\n")
    except KeyboardInterrupt:
        print("\nUser interrupt, closing...")
        if 'cr5' in locals():
            cr5.stop()
        if cr5.key_listener:
            cr5.key_listener.stop()