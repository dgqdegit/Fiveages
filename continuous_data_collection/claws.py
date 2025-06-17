"""

from pymodbus.client import ModbusTcpClient  

# 创建 Modbus TCP 客户端  
client = ModbusTcpClient('192.168.201.1', port=502)  # 请替换为实际的 IP 地址和端口  
client.connect()  

unit_id = 2  # 设备单元 ID  
read_address = 285  # 要读取的寄存器地址  

# 读取保持寄存器  
read_response = client.read_holding_registers(read_address, count=1, unit=unit_id)  # 只读取一个寄存器  
if not read_response.isError():  
    print(f"Read Holding Register at {read_address}: {read_response.registers[0]}")  
else:  
    print(f"Error reading holding register: {read_response}")  

# 关闭 Modbus 连接  
client.close()

"""

from pymodbus.client import ModbusTcpClient  

def read_modbus_holding_register(ip_address: str, port: int, unit_id: int, register_address: int) -> tuple:  
    """Read a holding register from a Modbus TCP server.  

    Args:  
        ip_address (str): The IP address of the Modbus server.  
        port (int): The TCP port number.  
        unit_id (int): The unit ID of the Modbus device.  
        register_address (int): The address of the holding register to read.  

    Returns:  
        tuple: A tuple containing a boolean flag indicating success, the value read (int) or an error message (str).  
    """  
    # 创建 Modbus TCP 客户端  
    client = ModbusTcpClient(ip_address, port)  
    
    # 尝试连接  
    if not client.connect():  
        return False, "Failed to connect to Modbus server."  

    # 读取保持寄存器  
    read_response = client.read_holding_registers(register_address, count=1, unit=unit_id)  

    # 关闭 Modbus 连接  
    client.close()  

    # 检查读取是否成功  
    if not read_response.isError():  
        return True, read_response.registers[0]  
    else:  
        return False, str(read_response)  

# 示例用法  
if __name__ == "__main__":  
    result_flag, result_value = read_modbus_holding_register('192.168.201.1', 502, 2, 258)  # 替换为实际的参数  
    if result_flag:  
        print(f"Read Holding Register Value: {result_value}")  
    else:  
        print(f"Error: {result_value}")