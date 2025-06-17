from pymodbus.client import ModbusTcpClient  

# 创建 Modbus TCP 客户端  
client = ModbusTcpClient('192.168.201.1', port=502)  # 请替换为实际的 IP 地址和端口  
client.connect()  

# 读取保持寄存器  
read_address = 258  # 要读取的地址  
num_registers = 2   # 读取的寄存器数量（根据需要进行调整）  
read_response = client.read_holding_registers(read_address, count=num_registers)  

# 检查是否读取成功  
if not read_response.isError():  
    print(f"读取保持寄存器: {read_response.registers}")  
else:  
    print(f"读取保持寄存器时出错: {read_response}")  

# 关闭 Modbus 连接  
client.close()