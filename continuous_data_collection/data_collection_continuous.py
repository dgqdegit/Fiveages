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
from pymodbus.client import ModbusTcpClient  # Add Modbus client  

class CR5Realtime:  
    def __init__(self, ip="192.168.201.1", port=30004, frequency=None):  
        self.ip = ip  
        self.port = port  
        self.running = False  
        self.data_queue = Queue(maxsize=100)  
        self.callbacks = []  
        self.realsense_camera = self._init_realsense_camera()  # Initialize the RealSense camera  
        self.zed_camera = self._init_zed_camera()  # Initialize the ZED camera  
        self.last_pose = None  # To store the last saved TCP pose  
        self.dragging = False  # Flag to indicate dragging state  
        self.claw_status = 0  # Initial claw state  
        self.claw_status_flag = None  # Initialize the status change flag  
        self.save_lock = threading.Lock()  # Thread lock for shared resources  

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

        # Create a base directory with timestamp for storing files  
        self.timestamp_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_t_wbl"  
        self.base_save_path = os.path.join(os.getcwd(), 'data', 'test_wbl', self.timestamp_dir_name)  
        os.makedirs(self.base_save_path, exist_ok=True)  
        os.makedirs(os.path.join(self.base_save_path, "realsense_rgb"), exist_ok=True)  
        os.makedirs(os.path.join(self.base_save_path, "zed_rgb"), exist_ok=True)  
        os.makedirs(os.path.join(self.base_save_path, "zed_depth"), exist_ok=True)  

        # Initialize the pose file  
        self.pose_file_path = os.path.join(self.base_save_path, "pose.txt")  
        self.extrinsics_file_path = os.path.join(self.base_save_path, "extrinsics.txt")  # Placeholder for extrinsics file  
        with open(self.pose_file_path, 'w') as f:  
            f.write("Timestamp Position (X, Y, Z) Orientation (Rx, Ry, Rz) Claw Status\n")  # Updated header for pose information  
        with open(self.extrinsics_file_path, 'w') as f:  
            f.write("Extrinsics information placeholder\n")  # Placeholder for extrinsics information  

    def _init_realsense_camera(self):  
        try:  
            pipeline = rs.pipeline()  
            config = rs.config()  
            serial_number = "327122077302"  # Replace with actual camera serial number  
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

    def _init_zed_camera(self):  
        try:  
            zed = sl.Camera()  
            init_params = sl.InitParameters()  
            init_params.camera_resolution = sl.RESOLUTION.HD1080  
            init_params.camera_fps = 30  
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL  
            init_params.coordinate_units = sl.UNIT.METER  
            if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:  
                raise RuntimeError("Failed to initialize ZED camera")  
            print("ZED camera initialized successfully.")  
            return zed  
        except Exception as e:  
            print(f"Failed to initialize ZED camera: {e}")  
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
            tcp_actual = struct.unpack('<6d', data[624:672])  # Get TCP pose  
            robot_mode = struct.unpack('<Q', data[24:32])[0]  

            is_dragging = robot_mode == 7

            # Read the claw status via Modbus  
            claw_status_flag, claw_status = self._read_claw_status()  
            self._update_claw_status(claw_status)  # Update claw status change  

            if is_dragging != self.dragging:  
                if is_dragging:  
                    print("Detected operation, start recording...")  
                    self.dragging = True  
                    self.last_pose = None  
                else:  
                    print("Operation stopped, stop recording...")  
                    self.dragging = False  

            if self.dragging:  
                current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  
                self._save_data_and_image(tcp_actual, current_timestamp)  

        except Exception as e:  
            print(f"Error processing data: {e}")  

    def _update_claw_status(self, new_status):  
        if new_status != self.claw_status:  
            self.claw_status_flag = 1  # Set flag if status changes  
            self.claw_status = new_status  
        else:  
            self.claw_status_flag = None  # Set flag to None if no change  

    def _read_claw_status(self) -> tuple:  
        """Read the claw status via Modbus."""  
        client = ModbusTcpClient(self.ip, port=502)  # Modbus connection to robot IP  
        if not client.connect():  
            return False, "Failed to connect to Modbus server."  

        try:  
            unit_id = 2  # Device unit ID  
            read_address = 258  # Register address for claw status  
            read_response = client.read_holding_registers(read_address, count=1, unit=unit_id)  

            if not read_response.isError():  
                return True, read_response.registers[0]  # Return flag indicating success and claw status  
            else:  
                return False, str(read_response)  # Return error message if reading fails  
        except Exception as err:  
            print(f"Error reading claw status: {err}")  
            return False, err  
        finally:  
            client.close()  # Ensure the Modbus connection is closed    

    def _save_data_and_image(self, tcp_actual, timestamp):  
        with self.save_lock:  
            try:  
                # Construct pose line without claw status if claw_status_flag is None  
                if self.claw_status_flag is not None:  
                    pose_line = f"{timestamp} " + " ".join(f"{pos:.4f}" for pos in tcp_actual) + f" {self.claw_status}\n"  
                else:  
                    pose_line = f"{timestamp} " + " ".join(f"{pos:.4f}" for pos in tcp_actual) + "\n"  # No claw status  
                
                with open(self.pose_file_path, 'a') as f:  
                    f.write(pose_line)  # Save TCP pose  

                # Capture and save images  
                self._capture_and_save_realsense_image(timestamp)  
                self._capture_and_save_zed_image(timestamp)  
                print(f"Data and images saved at timestamp: {timestamp}")  

            except Exception as e:  
                print(f"Error saving data and images: {e}")  

    def _capture_and_save_realsense_image(self, timestamp):  
        if not self.realsense_camera:  
            print("RealSense camera not initialized, cannot capture image.")  
            return  

        try:  
            frames = self.realsense_camera['pipeline'].wait_for_frames()  
            color_frame = frames.get_color_frame()  
            color_image = np.asanyarray(color_frame.get_data())  

            save_path = os.path.join(self.base_save_path, "realsense_rgb")  
            image_count = len(os.listdir(save_path))  
            image_path = os.path.join(save_path, f"{image_count + 0}_{timestamp}.png")  
            cv2.imwrite(image_path, color_image)  
            print(f"RealSense image saved to: {image_path}")  

        except Exception as e:  
            print(f"Error capturing and saving RealSense image: {e}")  

    def _capture_and_save_zed_image(self, timestamp):  
        if not self.zed_camera:  
            print("ZED camera not initialized, cannot capture image.")  
            return  

        try:  
            frame_rgb = sl.Mat()  
            frame_depth = sl.Mat()  
            if self.zed_camera.grab() == sl.ERROR_CODE.SUCCESS:  
                # Capture RGB Image  
                self.zed_camera.retrieve_image(frame_rgb, sl.VIEW.LEFT)  
                
                img_rgb_np = frame_rgb.get_data()  
                
                # Save RGB Image  
                save_path = os.path.join(self.base_save_path, "zed_rgb")  
                image_count = len(os.listdir(save_path))  
                image_path_rgb = os.path.join(save_path, f"{image_count + 0}_{timestamp}.png")  
                cv2.imwrite(image_path_rgb, img_rgb_np)  
                print(f"ZED RGB image saved to: {image_path_rgb}")  

                # Retrieve Depth Image  
                self.zed_camera.retrieve_measure(frame_depth, sl.MEASURE.XYZRGBA)  
                depth_data_np = frame_depth.get_data()  

                # Save Depth Image  
                save_path_depth = os.path.join(self.base_save_path, "zed_depth")  
                depth_image_count = len(os.listdir(save_path_depth))  
                depth_npy_path = os.path.join(save_path_depth, f"{depth_image_count + 0}_{timestamp}.npy")  
                np.save(depth_npy_path, depth_data_np)  
                print(f"ZED depth image saved to: {depth_npy_path}")  
            else:  
                print("Failed to capture ZED camera image.")  
        except Exception as e:  
            print(f"Error capturing and saving ZED images: {e}")  

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
        if run_time > 0:  
            actual_freq = self._stats['packets_processed'] / run_time  
        else:  
            actual_freq = 0  

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
            self.realsense_camera['pipeline'].stop()  # Stop RealSense camera  
        if self.zed_camera:  
            self.zed_camera.close()  # Stop ZED camera  
        print("CR5 data collection has stopped, all cameras have been closed.")  

# Usage example  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='CR5 robotic arm real-time data monitoring and dual camera image acquisition')  
    parser.add_argument('--ip', default='192.168.201.1', help='Robot arm IP address')  
    parser.add_argument('--port', type=int, default=30004, help='Data port (30004/30005/30006)')  
    parser.add_argument('--freq', type=int, default=10, help='Data processing frequency (Hz), default to 10Hz')  
    args = parser.parse_args()  

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
        cr5 = CR5Realtime(ip=args.ip, port=args.port, frequency=args.freq)  

        def pose_callback(data):  
            joints = data['joint_actual']  
            j_deg = [j * 57.2958 for j in joints]  

            cart = data['tcp_actual']  
            mode = data['robot_mode']  
            mode_str = ROBOT_MODES.get(mode, f"Unknown ({mode})")  

            print(f"Robot Status: {mode_str} | "  
                  f"Position: X:{cart[0]:.1f} Y:{cart[1]:.1f} Z:{cart[2]:.1f}mm | "  
                  f"Orientation: Rx:{cart[3]:.1f} Ry:{cart[4]:.1f} Rz:{cart[5]:.1f}° | "  
                  f"Joint Angles: [{', '.join([f'{j:.1f}°' for j in j_deg])}]")  

        cr5.register_callback(pose_callback)  

        cr5.start()  

        print("Press Ctrl+C to stop collection...")  
        while True:  
            time.sleep(5)  
            stats = cr5.get_stats()  
            print("\n--- Performance Statistics ---")  
            print(f"Uptime: {stats['uptime']}")  
            print(f"Actual Frequency: {stats['actual_frequency']}")  
            print(f"Current Latency: {stats['current_latency']}")  
            print(f"Maximum Latency: {stats['max_latency']}")  
            print("------------------------\n")  

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