import random
import socket
import re
import time
import os
from datetime import datetime
from pynput import keyboard
class Server():
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.baudrate = 115200
        self.modbus = None
        self.modbusRTU = None
        self.timestamp = None
        # self.init_com()
    def init_com(self):
        self.modbus = self.get_id(self.modbus)
        self.modbusRTU = self.get_id(self.modbusRTU)

    def get_id(self, response):
        match = re.search(r'\{(.+?)\}', response)

        if match:
            numbers_str = match.group(1)
            return int(numbers_str)
        
class Point():
    def __init__(self, name:str, position: dict):
        self.name = name
        self.position = position
        self.timestamp = None
        self.position_str = None
        self.claw = None
        self.position_to_string()
    def get_timestamp(self):
        dt = datetime.now()
        micro = dt.microsecond // 1000
        self.timestamp = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}")
        return self.timestamp
    def position_to_string(self):
        if self.position == None:
            return None
        self.position_str = f"{{{self.position['x']:.4f},{self.position['y']:.4f},{self.position['z']:.4f},\
            {self.position['rx']},{self.position['ry']},{self.position['rz']}}}"
        return self.position_str
start_motion = False  
def main():
    global start_motion
    # 创建一个 TCP/IP 套接字
    server = Server('192.168.201.1', 29999)
    # 连接到 DoBot 机械臂的 Dashboard 端口 (29999)
    server.sock.connect((server.ip, server.port))
    
    # 初始化机器人
    initialize_robot(server.sock)
    server_modbus = 'ModbusCreate("192.168.201.1", 502,2)'
    server_modbusrtu = 'ModbusRTUCreate(1, 115200, "N", 8, 1)'
    server.modbus = send_modbus_command(server.sock, server_modbus)
    server.modbusRTU = send_modbus_command(server.sock,server_modbusrtu)
    server.init_com()
    p1 = Point('p1', {'x': 173.2972, 'y': -130.0664, 'z': 167.6071, 'rx': 91.3785, 'ry': -0.9884, 'rz': -67.1547})
    p2 = Point('p2', {'x': 168.5996, 'y': -223.9824, 'z': 60, 'rx': 90.1654, 'ry': -3.2263, 'rz': -59.2746})
    p3 = Point('p3', {'x': -77.1242, 'y': -610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p4 = Point('p4', {'x': -187.2389, 'y':-610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p5 = Point('p5', {'x': -77.1242, 'y': -610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p6 = Point('p6', {'x': 168.5996, 'y': -223.9824, 'z': 60, 'rx': 90.1654, 'ry': 3.2263, 'rz': -59.2746})
    p7 = Point('p7', {'x': 173.2972, 'y': -130.0664, 'z': 167.6071, 'rx': 91.3785, 'ry': -0.9884, 'rz': -67.1547})
    point_list = [p1, p2, p3, p4, p5, p6, p7]
    def on_key_press(key):
        global start_motion
        if key == keyboard.KeyCode.from_char('q'):
            start_motion = True
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    try:
        while True:
            # 定义 p2 到 p6 的固定点位

            # 随机生成 p1 的 x 和 y 坐标，并保留 6 位小数
            # p2_x = round(random.uniform(100, 150), 6)
            # p2_y = round(random.uniform(-150, -100), 6)
            # point_list[1].position["x"] = p2_x
            # point_list[1].position["y"] = p2_y
            # point_list[1].position_to_string()
            # print(f"p2 的随机坐标: x={p2_x}, y={p2_y}")
            
            # 将点位转换为字符串格式，并保留 6 位小数

            # p1_str = f"{{{p1['x']:.4f},{p1['y']:.4f},{p1['z']:.4f},{p1['rx']},{p1['ry']},{p1['rz']}}}"
            # p2_str = f"{{{p2['x']:.4f},{p2['y']:.4f},{p2['z']:.4f},{p2['rx']},{p2['ry']},{p2['rz']}}}"
            # p3_str = f"{{{p3['x']:.4f},{p3['y']:.4f},{p3['z']:.4f},{p3['rx']},{p3['ry']},{p3['rz']}}}"
            # p4_str = f"{{{p4['x']:.4f},{p4['y']:.4f},{p4['z']:.4f},{p4['rx']},{p4['ry']},{p4['rz']}}}"
            # p5_str = f"{{{p5['x']:.4f},{p5['y']:.4f},{p5['z']:.4f},{p5['rx']},{p5['ry']},{p5['rz']}}}"
            # p6_str = f"{{{p6['x']:.4f},{p6['y']:.4f},{p6['z']:.4f},{p6['rx']},{p6['ry']},{p6['rz']}}}"
            # p7_str = f"{{{p7['x']:.4f},{p7['y']:.4f},{p7['z']:.4f},{p7['rx']},{p7['ry']},{p7['rz']}}}"

            # 定义一个标志变量，用于判断是否开始运动
            
            
            # 定义一个回调函数，用于监听键盘输入

            
            # 注册键盘监听
            # keyboard.on_press(on_key_press)
            
            # 提示用户按下 'a' 键开始运动
            print("按下 'q' 键开始机械臂运动...")
            
            # 等待用户按下 'a' 键
            while not start_motion:
                pass
            
            print("开始机械臂运动...")
            
            # # 依次发送关节运动指令
            dt = datetime.now()
            micro = dt.microsecond // 1000
            timestamp_start = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}_r_wbl")
            # send_movj_command(server.sock, p1.position_str)
            # wait_and_prompt(server.sock, p1)
            # send_movj_command(server.sock, p2.position_str)
            # wait_and_prompt(server.sock, p2)
            # claws_control(server.sock, 0, server.modbusRTU, p2)
            # wait_and_prompt(server.sock, p2)
            # send_movj_command(server.sock, p3.position_str)
            # wait_and_prompt(server.sock, p3)
            # send_movj_command(server.sock, p4.position_str)
            # wait_and_prompt(server.sock, p4)
            # claws_control(server.sock, 1, server.modbusRTU, p4)
            # wait_and_prompt(server.sock, p4)
            # send_movj_command(server.sock, p5.position_str)
            # wait_and_prompt(server.sock, p5)
            # send_movj_command(server.sock, p6.position_str)
            # wait_and_prompt(server.sock, p6)
            # send_movj_command(server.sock, p7.position_str)
            # wait_and_prompt(server.sock, p7)
            # print("机械臂运动完成。")
            # folder_path = timestamp_start
            # os.makedirs(folder_path, exist_ok=True)  # 创建文件夹
            # with open(os.path.join(folder_path, 'pose.txt'), 'w') as f:
            #     last_claw = 0
            #     for point in [p1, p2, p3, p4, p5, p6, p7]:
            #         now_claw = point.claw
            #         line = (
            #             f"{point.timestamp} "
            #             f"{point.position['x']:.6f} {point.position['y']:.6f} {point.position['z']:.6f} "
            #             f"{point.position['rx']:.6f} {point.position['ry']:.6f} {point.position['rz']:.6f} "
            #         )
            #         print(line)
            #         f.write(line)
            #         if now_claw != last_claw:
            #             f.write(f"{now_claw}")
            #         f.write(f"\n")
            replay_motion_trajectory(server.sock, server.modbusRTU, timestamp_start)
            start_motion = False
            # # 取消键盘监听，避免重复触发
            # keyboard.unhook_all()
            
            # # 重新注册键盘监听
            # keyboard.on_press(on_key_press)
            
    finally:
        # 关闭套接字连接
        send_modbus_command(server.sock, f'Modbusclose({server.modbus})')
        send_modbus_command(server.sock, f'Modbusclose({server.modbusRTU})')
        server.sock.close()
        listener.stop()
# 在main函数外新增辅助函数
def find_latest_timestamp_folder(folder_path):
    dir_list = os.listdir(folder_path)
    timestamp_folders = []
    pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_r_wbl$'# 匹配时间戳格式
    for name in dir_list:
        if os.path.isdir(os.path.join(folder_path, name)) and re.match(pattern, name):
            timestamp_folders.append(name)
    
    if not timestamp_folders:
        return None
    
    # 按时间戳排序，取最新文件夹
    timestamp_folders = sorted(
        timestamp_folders,
        key=lambda x: datetime.strptime(x[:19], "%Y-%m-%d_%H-%M-%S"),
        reverse=True
    )
    latest_folder = timestamp_folders[0]
    return latest_folder

def wait_and_prompt(sock, point=None, state = True, replay = False):
    while get_status(sock) != 5:
        time.sleep(0.1)
    # dt = datetime.now()
    # micro = dt.microsecond // 1000
    # timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S-{micro:03d}")
    if not replay:
        point.get_timestamp()

    if state:
        user_input = input("机械臂空闲，输入'y'继续下一动作：")
        while user_input.lower() != 'y':
            print("输入无效，请输入'y'确认继续！")
            user_input = input("机械臂空闲，输入'y'继续下一动作：")

ptest = Point('ptest', None)
#复现运动轨迹
def replay_motion_trajectory(sock, modbus, timestamp):
    trajectory_points = []
    cnt = 0
    print("检测到轨迹文件，输入'y'确认执行复现，其他输入取消：")
    user_choice = input().lower()
    global ptest
    if user_choice != 'y':
        print("取消轨迹复现，继续等待'a'键...")
        return 0
    try:
        # folder_path = timestamp
        folder_path = find_latest_timestamp_folder('data/right_wbl/')
        with open(os.path.join('data/right_wbl/', folder_path, 'pose.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            if len(parts) >=7:  # 至少包含时间戳+6个坐标参数
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    rx = float(parts[4])
                    ry = float(parts[5])
                    rz = float(parts[6])
                    trajectory_points.append({
                        'x':x, 'y':y, 'z':z,
                        'rx':rx, 'ry':ry, 'rz':rz
                    })
                except ValueError:
                    print(f"警告：无效数据行：{line}")

        if len(trajectory_points) >=7:
            print("开始轨迹复现：")
            for point in trajectory_points:
                point_str = f"{{{point['x']:.4f},{point['y']:.4f},{point['z']:.4f},{point['rx']},{point['ry']},{point['rz']}}}"
                send_movj_command(sock, point_str)
                wait_and_prompt(sock, state = False, replay= True)
                if cnt == 1:
                    claws_control(sock, 0, modbus, ptest)
                elif cnt == 3:
                    claws_control(sock, 1, modbus, ptest)
                cnt += 1
            print("轨迹复现完成！")
        else:
            print("轨迹点不足，未执行复现")
    except FileNotFoundError:
        print("轨迹文件未找到，无法复现轨迹")
def initialize_robot(sock):


    send_command(sock, "PowerOn()")
    

    time.sleep(1)

    # 使能机械臂，不设置负载和偏心参数
    send_command(sock, "EnableRobot()")
    
    # 清除机器人报警
    send_command(sock, "ClearError()")

def send_command(sock, command):
    # 发送指令
    sock.sendall(f"{command}\n".encode('utf-8'))
    
    # 接收响应
    response = sock.recv(1024).decode('utf-8')
    
    # 打印响应
    print(f"Command: {command}")
    print(f"Response: {response}")
    return response

def send_movj_command(sock, point):
    # 发送 MovJ 指令
    command = f"MovJ(pose={point},a=30,v=30)"
    send_command(sock, command)

def send_modbus_command(sock, command):
    # 发送 ModBus 指令
    command = f"{command}"
    return send_command(sock, command)
def claws_send_command(sock, id, num1, num2, num3):
    command = f'SetHoldRegs({id}, {num1}, {num2}, {{{num3}}}, "U16")'
    send_command(sock, command)
def claws_control(sock, status, id, point):
    if status: # 打开机械臂爪子
        claws_send_command(sock, id, 258, 1, 0)
        claws_send_command(sock, id, 259, 1, 1)
        claws_send_command(sock, id, 264, 1, 1)
        claws_send_command(sock, 0, 258, 1, 0)
        time.sleep(1)
        point.claw = 0
    else: # 关闭机械臂爪子
        claws_send_command(sock, id, 258, 1, 1)
        claws_send_command(sock, id, 259, 1, 0)
        claws_send_command(sock, id, 264, 1, 1)
        claws_send_command(sock, 0, 258, 1, 1) 
        time.sleep(1)
        point.claw = 1
def get_status(sock):
    command = "RobotMode()"
    response = send_command(sock, command)
    status_code = int(response.split(',')[1][1])
    return status_code

if __name__ == "__main__":
    main()