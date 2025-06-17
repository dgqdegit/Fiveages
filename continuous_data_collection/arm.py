import socket
import time
import logging
import re
import struct
import threading


# class DobotClient:
class Arm:
    """
    Dobot机器人TCP/IP控制客户端
    支持通过Dashboard端口(29999)发送控制指令
    支持通过反馈端口(30004/30005)监控机器人状态 
    """

    # 机器人状态码
    ROBOT_MODE_INIT = 1  # 初始化状态
    ROBOT_MODE_BRAKE_OPEN = 2  # 有任意关节的抱闸松开
    ROBOT_MODE_POWEROFF = 3  # 机械臂下电状态
    ROBOT_MODE_DISABLED = 4  # 未使能（无抱闸松开）
    ROBOT_MODE_ENABLED = 5  # 使能状态
    ROBOT_MODE_BACKDRIVE = 6  # 拖拽状态
    ROBOT_MODE_RUNNING = 7  # 运行状态
    ROBOT_MODE_PAUSED = 8  # 暂停状态
    ROBOT_MODE_ERROR = 9  # 错误状态

    def __init__(self, ip="192.168.201.1", port=29999, feedback_port=30004, timeout=10):
        """
        初始化Dobot客户端

        Args:
            ip (str): 机器人控制器IP地址
            port (int): 控制端口，默认29999(Dashboard)
            feedback_port (int): 反馈端口，默认30004(8ms反馈)
            timeout (int): 连接超时时间(秒)
        """
        self.ip = ip
        self.port = port
        self.feedback_port = feedback_port
        self.timeout = timeout
        self.socket = None
        self.feedback_socket = None
        self.connected = False
        self.feedback_connected = False
        self.logger = self._setup_logger()

        # 机器人状态
        self.robot_mode = 0
        self.is_moving = False

        # 反馈数据监控线程
        self.feedback_thread = None
        self.stop_feedback = False

    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger("Arm")
        logger.setLevel(logging.INFO)

        # 检查是否已有处理器，避免重复添加
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def connect(self):
        """连接到机器人控制器"""
        try:
            # 连接到Dashboard端口
            self.socket = socket.socket()
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.ip, self.port))
            self.connected = True
            self.logger.info(f"成功连接到Dobot机器人Dashboard端口 {self.ip}:{self.port}")

            # 连接到反馈端口
            self.connect_feedback()

            return True
        except Exception as e:
            self.logger.error(f"连接失败: {str(e)}")
            self.connected = False
            return False

    def connect_feedback(self):
        """连接到反馈端口"""
        try:
            self.feedback_socket = socket.socket()
            self.feedback_socket.settimeout(self.timeout)
            self.feedback_socket.connect((self.ip, self.feedback_port))
            self.feedback_connected = True
            self.logger.info(f"成功连接到Dobot机器人反馈端口 {self.ip}:{self.feedback_port}")

            # 启动反馈数据监控线程
            self.stop_feedback = False
            self.feedback_thread = threading.Thread(target=self._feedback_handler, daemon=True)
            self.feedback_thread.start()

            return True
        except Exception as e:
            self.logger.error(f"连接反馈端口失败: {str(e)}")
            self.feedback_connected = False
            return False

    def disconnect(self):
        """断开与机器人的连接"""
        # 停止反馈线程
        if self.feedback_thread and self.feedback_thread.is_alive():
            self.stop_feedback = True
            self.feedback_thread.join(timeout=2)

            # 关闭反馈socket
        if self.feedback_socket:
            try:
                self.feedback_socket.close()
            except:
                pass
            self.feedback_connected = False

            # 关闭控制socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.connected = False

        self.logger.info("已断开与机器人的连接")

    def _feedback_handler(self):
        """处理反馈数据的线程函数"""
        self.logger.info("反馈数据监控线程已启动")

        while not self.stop_feedback:
            try:
                # 接收反馈数据 (1440字节)
                data = self.feedback_socket.recv(1440)
                if not data or len(data) < 32:  # 至少要包含RobotMode数据
                    time.sleep(0.001)
                    continue

                    # 解析RobotMode (位于24-31字节)
                robot_mode = struct.unpack('<Q', data[24:32])[0]
                self.robot_mode = robot_mode

                # 解析是否在运动
                # 根据文档，通常可以通过运动状态位判断，这里简化为根据RobotMode判断
                # 实际使用时可能需要解析更多字段
                self.is_moving = (robot_mode == self.ROBOT_MODE_RUNNING)

            except socket.timeout:
                # 超时不是错误，继续尝试
                pass
            except Exception as e:
                self.logger.error(f"处理反馈数据出错: {str(e)}")
                time.sleep(0.1)  # 避免过于频繁的错误日志

    def send_command(self, command):
        """
        发送命令并接收响应

        Args:
            command (str): 要发送的命令

        Returns:
            tuple: (ErrorID, result) - 错误码和结果
        """
        if not self.connected:
            self.logger.error("未连接到机器人，请先调用connect()")
            return -1, "未连接到机器人"

        try:
            # 确保命令以换行符结束
            if not command.endswith('\n'):
                command += '\n'

                # 发送命令
            self.socket.send(command.encode())
            self.logger.debug(f"已发送: {command.strip()}")

            # 接收响应
            response = self.socket.recv(1024).decode()
            self.logger.debug(f"已接收: {response.strip()}")

            # 解析响应
            pattern = r"^(\-?\d+),\{(.*)\},(.*);$"
            match = re.match(pattern, response)

            if match:
                error_id = int(match.group(1))
                result = match.group(2)
                cmd_echo = match.group(3)

                if error_id != 0:
                    self.logger.warning(f"命令执行出错，错误码: {error_id}")

                return error_id, result
            else:
                self.logger.warning(f"无法解析响应: {response}")
                return -1, response

        except Exception as e:
            self.logger.error(f"发送命令出错: {str(e)}")
            return -1, str(e)

    def wait_for_motion_completion(self, timeout=60):
        """
        等待机器人运动完成

        Args:
            timeout (int): 超时时间(秒)

        Returns:
            bool: 是否成功等待运动完成
        """
        start_time = time.time()

        # 首先等待机器人进入运动状态
        # 有时命令发送后机器人需要一点时间才开始运动
        wait_start = time.time()
        while time.time() - wait_start < 2:  # 最多等待2秒
            if self.is_moving or self.robot_mode == self.ROBOT_MODE_RUNNING:
                break
            time.sleep(0.1)

            # 等待机器人停止运动
        self.logger.info("等待机器人运动完成...")
        while time.time() - start_time < timeout:
            # 如果机器人不再处于运动状态
            if not self.is_moving and self.robot_mode != self.ROBOT_MODE_RUNNING:
                self.logger.info("机器人运动已完成")
                return True

                # 如果机器人处于错误状态
            if self.robot_mode == self.ROBOT_MODE_ERROR:
                self.logger.error("机器人进入错误状态，运动被中断")
                return False

            time.sleep(0.2)  # 降低轮询频率

        self.logger.warning(f"等待机器人运动完成超时({timeout}秒)")
        return False

        # 控制相关方法

    def power_on(self):
        """机器人上电"""
        return self.send_command("PowerOn()")

    def enable_robot(self, load=None, center_x=None, center_y=None, center_z=None, is_check=None):
        """
        使能机器人

        Args:
            load (float, optional): 负载重量(kg)
            center_x (float, optional): X方向偏心距离(mm)
            center_y (float, optional): Y方向偏心距离(mm)
            center_z (float, optional): Z方向偏心距离(mm)
            is_check (int, optional): 是否检查负载(1:检查, 0:不检查)
        """
        cmd = "EnableRobot("
        params = []

        if load is not None:
            params.append(str(load))
            if center_x is not None and center_y is not None and center_z is not None:
                params.extend([str(center_x), str(center_y), str(center_z)])
                if is_check is not None:
                    params.append(str(is_check))

        cmd += ",".join(params) + ")"
        return self.send_command(cmd)

    def disable_robot(self):
        """下使能机器人"""
        return self.send_command("DisableRobot()")

    def clear_error(self):
        """清除错误"""
        return self.send_command("ClearError()")

    def stop(self):
        """停止机器人运动"""
        return self.send_command("Stop()")

    def emergency_stop(self):
        """紧急停止"""
        return self.send_command("EmergencyStop()")

        # 运动相关方法

    def mov_j(self, pose, user=None, tool=None, a=None, v=None, cp=None, wait=True):
        """
        从当前位置以关节运动方式运动至目标点

        Args:
            pose (str): 目标点，支持关节变量或位姿变量，例如"joint={0,0,0,0,0,0}"或"pose={-500,100,200,150,0,90}"
            user (int, optional): 用户坐标系
            tool (int, optional): 工具坐标系
            a (int, optional): 加速度比例，范围(0,100]
            v (int, optional): 速度比例，范围(0,100]
            cp (int, optional): 平滑过渡比例，范围[0,100]
            wait (bool, optional): 是否等待运动完成，默认为True

        Returns:
            tuple: (ErrorID, result) 如果wait=True，还会等待运动完成
        """
        cmd = f"MovJ({pose}"

        if user is not None:
            cmd += f", user={user}"
        if tool is not None:
            cmd += f", tool={tool}"
        if a is not None:
            cmd += f", a={a}"
        if v is not None:
            cmd += f", v={v}"
        if cp is not None:
            cmd += f", cp={cp}"

        cmd += ")"
        error_id, result = self.send_command(cmd)

        # 如果指令成功发送且需要等待运动完成
        if error_id == 0 and wait:
            self.wait_for_motion_completion()

        return error_id, result

    def mov_l(self, pose, user=None, tool=None, a=None, v=None, cp=None, r=None, speed=None, wait=True):
        """
        从当前位置以直线运动方式运动至目标点

        Args:
            pose (str): 目标点，支持关节变量或位姿变量
            user (int, optional): 用户坐标系
            tool (int, optional): 工具坐标系
            a (int, optional): 加速度比例，范围(0,100]
            v (int, optional): 速度比例，范围(0,100]
            cp (int, optional): 平滑过渡比例，范围[0,100]
            r (int, optional): 平滑过渡半径(mm)
            speed (int, optional): 绝对速度(mm/s)
            wait (bool, optional): 是否等待运动完成，默认为True

        Returns:
            tuple: (ErrorID, result) 如果wait=True，还会等待运动完成
        """
        cmd = f"MovL({pose}"

        if user is not None:
            cmd += f", user={user}"
        if tool is not None:
            cmd += f", tool={tool}"
        if a is not None:
            cmd += f", a={a}"
        if v is not None and speed is None:
            cmd += f", v={v}"
        if speed is not None:
            cmd += f", speed={speed}"
        if cp is not None and r is None:
            cmd += f", cp={cp}"
        if r is not None:
            cmd += f", r={r}"

        cmd += ")"
        error_id, result = self.send_command(cmd)

        # 如果指令成功发送且需要等待运动完成
        if error_id == 0 and wait:
            self.wait_for_motion_completion()

        return error_id, result

    def servo_j(self, joint, t=0.1, aheadtime=50, gain=500, wait=True):
        """
        基于关节空间的动态跟随命令

        Args:
            joint (str): 目标关节角度，例如"{0,0,0,0,0,0}"
            t (float, optional): 运行时间(s)，范围[0.02,3600.0]
            aheadtime (float, optional): 提前量，范围[20.0,100.0]
            gain (float, optional): 目标位置比例增益，范围[200.0,1000.0]
            wait (bool, optional): 是否等待运动完成，默认为True
        """
        cmd = f"ServoJ({joint}, t={t}, aheadtime={aheadtime}, gain={gain})"
        error_id, result = self.send_command(cmd)

        # 如果指令成功发送且需要等待运动完成
        if error_id == 0 and wait:
            self.wait_for_motion_completion()

        return error_id, result

    def move_jog(self, axis_id, wait=False):
        """
        点动机械臂

        Args:
            axis_id (str): 点动轴和方向，例如"J1+"或"X-"
            wait (bool, optional): 是否等待运动完成，默认为False（点动通常是持续的）
        """
        cmd = f"MoveJog({axis_id})"
        error_id, result = self.send_command(cmd)

        # 点动通常需要停止命令来终止，如果wait=True，则等待
        if error_id == 0 and wait:
            self.wait_for_motion_completion()

        return error_id, result

    def get_pose(self):
        """获取机械臂当前位姿"""
        return self.send_command("GetPose()")

    def get_angle(self):
        """获取机械臂当前关节角度"""
        return self.send_command("GetAngle()")

    def get_robot_mode(self):
        """获取机器人当前状态"""
        return self.send_command("RobotMode()")


def recover_from_emergency_stop(robot):
    """
    从急停状态恢复机器人

    Args:
        robot: Arm实例

    Returns:
        bool: 恢复是否成功
    """
    print("开始从急停状态恢复机器人...")

    # 步骤1: 确认急停开关已释放（物理操作，需要手动旋转急停按钮）
    input("请确认急停按钮已经释放（旋转急停按钮使其弹起），然后按Enter继续...")

    # 步骤2: 清除错误
    print("清除错误...")
    error_id, result = robot.clear_error()
    if error_id != 0:
        print(f"清除错误失败: {result}")
        return False
    print("错误已清除")

    # 步骤3: 重新上电
    print("机器人重新上电...")
    error_id, result = robot.power_on()
    if error_id != 0:
        print(f"上电失败: {result}")
        return False

        # 上电需要时间
    print("等待上电完成...")
    time.sleep(10)

    # 步骤4: 重新使能机器人
    print("使能机器人...")
    error_id, result = robot.enable_robot()
    if error_id != 0:
        print(f"使能失败: {result}")
        return False

        # 使能需要时间
    time.sleep(2)

    # 步骤5: 验证机器人状态
    error_id, mode = robot.get_robot_mode()
    print(f"机器人当前状态: {mode}")

    # 检查机器人是否恢复到正常状态
    if robot.robot_mode == robot.ROBOT_MODE_ENABLED:
        print("机器人已成功恢复到工作状态")
        return True
    else:
        print("机器人未能恢复到正常工作状态，可能需要进一步检查")
        return False


def demo():
    """演示程序"""
    # 替换为你的机器人IP地址
    robot_ip = "192.168.201.1"

    # 创建机器人控制客户端
    robot = Arm(robot_ip)

    try:
        # 连接到机器人
        if not robot.connect():
            print("无法连接到机器人，程序退出")
            return

            # 检查机器人当前状态
        error_id, mode = robot.get_robot_mode()
        print(f"机器人当前状态: {mode}")

        # 如果机器人处于急停状态，执行恢复流程
        if robot.robot_mode == robot.ROBOT_MODE_ERROR:
            print("检测到机器人处于错误状态，尝试清除错误...")
            robot.clear_error()
            time.sleep(1)

            # 清除可能存在的错误
        robot.clear_error()

        # 机器人上电
        print("机器人上电中...")
        robot.power_on()
        # 上电需要时间，等待10秒
        print("等待上电完成...")
        time.sleep(5)

        # 使能机器人
        print("使能机器人...")
        robot.enable_robot()
        time.sleep(2)

        # 获取当前机器人状态
        # error_id, result = robot.get_robot_mode()
        # print(f"机器人当前状态: {result}")

        # 获取当前位置
        # 获取当前姿势，假设返回的姿势是 mm 的单位  
        # arm_pose_mm = robot.get_pose()[1]  # 当前位置，单位为毫米  
        # arm_pose_list = arm_pose_mm.split(',')    # 根据逗号分割  
        # arm_pose_mm = list(map(float, arm_pose_list))  # 转换为浮点数列表  
        # # 将前3个值转换为米  
        # x_m = arm_pose_mm[0]/ 1000.0  # 将 x 从 mm 转换为 m  
        # y_m = arm_pose_mm[1]/ 1000.0  # 将 y 从 mm 转换为 m  
        # z_m = arm_pose_mm[2] / 1000.0  # 将 z 从 mm 转换为 m  

        # # 获取旋转的角度（保持不变）  
        # roll = arm_pose_mm[3] # 确保为浮点数  
        # pitch = arm_pose_mm[4] # 确保为浮点数  
        # yaw = arm_pose_mm[5] # 确保为浮点数  

        # # 创建新的姿势列表，所有元素都是浮点数  
        # arm_pose_floats = [x_m, y_m, z_m, roll, pitch, yaw]  
        # print(arm_pose_floats)
        # print(type(arm_pose_floats))
        # print(arm_pose_floats[0])
        # print(type(arm_pose_floats[0]))

        # 输出新的姿势列表  
        # print("Arm Pose (as floats):", arm_pose_floats)


        error_id, result = robot.get_pose()
        print(f"当前位姿: {result}")

        # # 获取当前关节角度
        # error_id, result = robot.get_angle()
        # print(f"当前关节角度: {result}")
  


        
        # # 执行一个简单的关节运动，并等待完成
        # print("执行关节运动...")
        # robot.mov_j("joint={10,20,30,40,50,60}", v=20, wait=True)
        # print("关节运动已完成")

        # # 执行一个简单的直线运动，并等待完成
        # print("执行直线运动...")
        # robot.mov_l("pose={300,0,400,0,0,0}", v=30, wait=True)
        # print("直线运动已完成")

        # # 执行一系列点位运动
        # print("执行多点位运动...")
        # points = [  
        #     "pose={300, 100, 300, 0, 0, 0}",  # 初始位置  
        #     "pose={300, 150, 350, 0, 0, 30}",  
        #     "pose={250, 100, 300, 0, 0, 45}",  
        #     "pose={350, 100, 250, 0, 0, -30}",  
        #     # 添加更多位置，直到有15个不同的位置  
        #     "pose={300, 100, 400, 0, 0, 0}",  
        #     "pose={300, 80, 350, 0, 0, 20}",  
        #     "pose={250, 150, 300, 0, 0, 0}",  
        #     "pose={350, 150, 300, 0, 0, 0}",  
        #     "pose={300, 100, 250, 0, 0, 45}",  
        #     "pose={300, 100, 300, 0, 0, 30}",  
        #     "pose={300, 80, 300, 0, 0, 45}",  
        #     "pose={300, 120, 350, 0, 0, -30}",  
        #     "pose={250, 80, 300, 0, 0, 0}",  
        #     "pose={350, 80, 300, 0, 0, 0}",  
        #     "pose={300, 100, 300, 0, 0, -45}",  
        # ]  

        # for i, point in enumerate(points):
        #     print(f"移动到点 {i + 1}/{len(points)}")
        #     robot.mov_l(point, v=30, wait=True)
        #     print(f"已到达点 {i + 1}")
        # # 执行持续的多点位运动.....
        # print("执行持续的多点位运...")
        # points = [
        #     "pose={300,100,400,0,0,0}",
        #     "pose={300,100,300,0,0,0}",
        #     "pose={300,0,300,0,0,0}",
        #     "pose={300,0,400,0,0,0}"
        # ]

        # for i, point in enumerate(points):
        #     print(f"移动到点 {i + 1}/{len(points)}")
        #     robot.mov_l(point, v=30, wait=True)
        #     print(f"已到达点 {i + 1}")
        
        # 下使能机器人
        print("下使能机器人...")
        robot.disable_robot()

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
    finally:
        # 确保断开连接
        robot.disconnect()
        print("演示程序结束")


if __name__ == "__main__":
    import trimesh  
    import numpy as np  
    import math
    # 创建一个坐标轴的线段  

    robot=Arm()
    # 连接到机器人
    robot.connect()
        

    robot.power_on()
    # 上电需要时间，等待10秒
    print("等待上电完成...")
    time.sleep(5)

    # 使能机器人
    print("使能机器人...")
    robot.enable_robot()
    time.sleep(2)
    pose = robot.get_pose()
    print(pose)
    
