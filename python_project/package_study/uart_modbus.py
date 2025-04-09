import time
import serial
import serial.tools.list_ports

GLOBAL_CONSTANT_SLAVE_ADDR = b'\x01'   # 从机地址固定位01H

GLOBAL_CONSTANT_READ_COIL = b'\x01'   # 读线圈指令
GLOBAL_CONSTANT_READ_INPUT_COIL = b'\x02'   # 读输入线圈指令
GLOBAL_CONSTANT_READ_REG = b'\x03'   # 读出寄存器指令
GLOBAL_CONSTANT_READ_INPUT_REG = b'\x04'   # 读输入寄存器指令
GLOBAL_CONSTANT_WRITE_SINGLE_COIL = b'\x05'   # 写单个线圈指令
GLOBAL_CONSTANT_WRITE_SINGLE_REG = b'\x06'   # 写单个寄存器指令
GLOBAL_CONSTANT_WRITE_MULT_REG = b'\x10'   # 写多个寄存器指令
GLOBAL_CONSTANT_WRITE_MULT_COIL = b'\x0F'   # 写多个线圈指令

# 线圈基址地址
GLOBAL_CONSTANT_COIL_BASE = 0x00  # 0 ~ 4FFF
# 线圈地址映射
GLOBAL_CONSTANT_COIL_ADDR = {
    '相机五轴快速定位设定X坐标命令': 90,
    '相机五轴快速定位设定Y坐标命令': 91,
    '相机五轴快速定位设定Z坐标命令': 92,
    '相机五轴快速定位设定U坐标命令': 93,
    '相机五轴快速定位设定R坐标命令': 94,
    '相机五轴X轴移动+命令': 100,
    '相机五轴X轴移动-命令': 101,
    '相机五轴Y轴移动+命令': 102,
    '相机五轴Y轴移动-命令': 103,
    '相机五轴Z轴移动+命令': 104,
    '相机五轴Z轴移动-命令': 105,
    '相机五轴U轴移动+命令': 106,
    '相机五轴U轴移动-命令': 107,
    '相机五轴R轴移动+命令': 108,
    '相机五轴R轴移动-命令': 109,
    '相机五轴X轴回原点命令': 140,
    '相机五轴Y轴回原点命令': 141,
    '相机五轴Z轴回原点命令': 142,
    '相机五轴U轴回原点命令': 143,
    '相机五轴R轴回原点命令': 144,
    '机械手五轴快速定位设定X坐标命令': 10,
    '机械手五轴快速定位设定Y坐标命令': 96,
    '机械手五轴快速定位设定Z坐标命令': 97,
    '机械手五轴快速定位设定U坐标命令': 98,
    '机械手五轴快速定位设定R坐标命令': 99,
    '机械手五轴X轴移动+命令': 110,
    '机械手五轴X轴移动-命令': 111,
    '机械手五轴Y轴移动+命令': 112,
    '机械手五轴Y轴移动-命令': 113,
    '机械手五轴Z轴移动+命令': 114,
    '机械手五轴Z轴移动-命令': 115,
    '机械手五轴U轴移动+命令': 116,
    '机械手五轴U轴移动-命令': 117,
    '机械手五轴R轴移动+命令': 118,
    '机械手五轴R轴移动-命令': 119,
    '机械手五轴X轴回原点命令': 145,
    '机械手五轴Y轴回原点命令': 146,
    '机械手五轴Z轴回原点命令': 147,
    '机械手五轴U轴回原点命令': 148,
    '机械手五轴R轴回原点命令': 149
}

# 寄存器基址地址
GLOBAL_CONSTANT_REG_BASE = 0xA080  # A080 ~ B87F
# 寄存器地址映射
GLOBAL_CONSTANT_REG_ADDR = {
    '相机五轴当前显示x坐标': 200,
    '相机五轴当前显示Y坐标': 202,
    '相机五轴当前显示Z坐标': 204,
    '相机五轴当前显示U坐标': 206,
    '相机五轴当前显示R坐标': 208,
    '相机五轴设定X坐标': 400,
    '相机五轴设定Y坐标': 402,
    '相机五轴设定Z坐标': 404,
    '相机五轴设定U坐标': 406,
    '相机五轴设定R坐标': 408,
    '相机五轴设定手动X速度': 500,
    '相机五轴设定手动Y速度': 502,
    '相机五轴设定手动Z速度': 504,
    '相机五轴设定手动U速度': 506,
    '相机五轴设定手动R速度': 508,
    '机械手五轴当前显示X坐标': 210,
    '机械手五轴当前显示Y坐标': 212,
    '机械手五轴当前显示Z坐标': 214,
    '机械手五轴当前显示U坐标': 216,
    '机械手五轴当前显示R坐标': 218,
    '机械手五轴设定X坐标': 410,
    '机械手五轴设定Y坐标': 412,
    '机械手五轴设定Z坐标': 414,
    '机械手五轴设定U坐标': 416,
    '机械手五轴设定R坐标': 418,
    '机械手五轴设定手动X速度': 510,
    '机械手五轴设定手动Y速度': 512,
    '机械手五轴设定手动Z速度': 514,
    '机械手五轴设定手动U速度': 516,
    '机械手五轴设定手动R速度': 518
}


def get_serial_port_list():
    """ 获取可用串口设备信息 """
    ports_list = list(serial.tools.list_ports.comports())
    if len(ports_list) > 0:
        print("可用的串口设备如下：")
        for com_port in ports_list:
            print(list(com_port)[0], list(com_port)[1])


class UartTTL:
    """ 串口类, 用于通信 """

    def __init__(self, port: str):
        """ 初始化, 这里仅要求输入串口号 """
        self.uart = serial.Serial(  # 构建成功后,自动打开
            port=port,  # 端口
            baudrate=19200,  # 波特率
            bytesize=serial.EIGHTBITS,  # 数据位, 8
            parity=serial.PARITY_EVEN,  # 校验位, EVEN
            stopbits=serial.STOPBITS_ONE,  # 停止位, 1
            timeout=0.3,  # 读超时时间，支持小数
            write_timeout=None,  # 写超时时间
        )
        self.is_open()

    def is_open(self):
        """ 打印串口的开关信息 """
        if self.uart.isOpen():
            print(f"{self.uart.name} 已打开")
        else:
            print(f"{self.uart.name} 已关闭")

    def open(self):
        """ 打开串口 """
        self.uart.open()
        self.is_open()

    def close(self):
        """ 关闭串口 """
        self.uart.close()
        self.is_open()

    def send_data(self, data: str | bytes):
        """ 发送数据 """
        if isinstance(data, str):
            ret = self.uart.write(data.encode("ascii"))
        elif isinstance(data, bytes):
            ret = self.uart.write(data)
        else:
            print("ERROR! 数据类型错误!")
        print(f"发出{ret}个字节", data)

    def read_data(self):
        """ 接收数据 """
        data = self.uart.read(100)
        print(f"共接收字节数: {len(data)}")
        return data

    def _crc16(self, data: bytes) -> int:
        """ 计算CRC

        验证: data = b'\x01\x06\x00\x02\x13\x88'
        输出: \x25 \x5c
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc.to_bytes(2, 'little')

    def write_single_register(self, reg_name: str, data: int | bytes):
        """ 写单个寄存器 """
        time.sleep(0.01)
        write_data = GLOBAL_CONSTANT_SLAVE_ADDR
        write_data += GLOBAL_CONSTANT_WRITE_SINGLE_REG
        addr = GLOBAL_CONSTANT_REG_ADDR[reg_name]
        addr += GLOBAL_CONSTANT_REG_BASE
        write_data += addr.to_bytes(2, byteorder="big", signed=False)
        if isinstance(data, int):
            write_data += data.to_bytes(2, byteorder="big", signed=False)
        elif isinstance(data, bytes):
            write_data += data
        else:
            print("ERROR! 数据类型错误!")
        crc = self._crc16(write_data)
        write_data += crc
        self.send_data(write_data)
        time.sleep(0.01)

    def write_single_coil(self, reg_addr: str, data: str = "OFF"):
        """ 写单个线圈 """
        time.sleep(0.01)
        write_data = GLOBAL_CONSTANT_SLAVE_ADDR
        write_data += GLOBAL_CONSTANT_WRITE_SINGLE_COIL
        addr = GLOBAL_CONSTANT_COIL_ADDR[reg_addr]
        addr += GLOBAL_CONSTANT_COIL_BASE
        write_data += addr.to_bytes(2, byteorder="big", signed=False)
        if data == "ON":
            # 单个线圈写时， ON 为 00FFH OFF 为 0000H ；且数据内容是低字节数据在前，高字节数据
            write_data += bytes.fromhex('FF 00')
        elif data == "OFF":
            write_data += bytes.fromhex('00 00')
        else:
            print("ERROR! 数据指令错误!")
        crc = self._crc16(write_data)
        write_data += crc
        self.send_data(write_data)
        time.sleep(0.01)


if __name__ == "__main__":
    # 查看连接的串口设备
    get_serial_port_list()
    # 选择对应的串口名称
    uart = UartTTL("COM3")
    uart.write_single_coil("相机五轴X轴回原点命令", "ON")
    uart.close()
