from botarm import *

server = Server('192.168.5.2', 29999, '192.168.110.235', 12345)
# 连接到 DoBot 机械臂的 Dashboard 端口 (29999)
server.sock.connect((server.ip, server.port))
bot = DobotController(server.sock)
server.init_com(bot)

# 初始化机器人
bot._initialize(server)

bot.claws_control(1, server.modbusRTU)  #  打开夹爪
"""
如果夹爪没有反应，先将控制板上的设备模式切换刷新
"""


# initial_joint = (-276.2836, -34.9111, 131.7840, -6.6240, -89.9493, 0.4157)   #（适用于 方块放盘子任务 and 按压洗发水任务）

# initial_joint = (-240.0731, 13.7763, 145.8175, 22.9280, 152.6269, 4.4483)   #（适用于 放瓶子进微波炉任务）

initial_joint = (-323.2057, -9.0527, 127.3597, 60.5590, 40.2580, 94.3243)   #（适用于 抓取动物放进抽屉任务）

# initial_joint = (249.7275, -17.3400, -146.6193, -14.4953, 217.1621, 60.4939)   #（适用于  打开微波炉任务）

# initial_joint = (-275.1692, -14.6800, 112.9294, -10.4861, -93.2329, 1.996)   #（适用于  架子任务）
bot.control_movement('joint', initial_joint)