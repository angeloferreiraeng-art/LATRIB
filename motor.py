from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
from dataclasses import dataclass

import numpy as np
import math

# Modbus client params
MODBUS_PORT = f"COM{5}"
MODBUS_SLAVE_ID = 1
MODBUS_BAUD_RATE = 19200  # 115200
MODBUS_PARITY = "E"
MODBUS_STOPBITS = 1
MODBUS_BYTESIZE = 8

# Command values
START_CMD = 0x0001
STOP_CMD = 0x0000


# Controller memory addresses
ADDR_SPEED_VAL = 0x0600 + 3  # 1540-1 : RPM setpoint (int16)
ADDR_CTRL_MODE = 0x0200 + 0  # control mode (speed/pos/torque)
ADDR_DI1_FN = 0x0300 + 2  # DI1 func (disable)
ADDR_SPEED_SRC_A = 0x0600 + 0  # main speed source A
ADDR_SPEED_SRC_B = 0x0600 + 1  # aux speed source B
ADDR_SPEED_SRC_SEL = 0x0600 + 2  # speed command source
ADDR_VDI_ENABLE = 0x0C00 + 9  # comm VDI enable
ADDR_VDI_LEVEL = 0x3100 + 0  # VDI virtual
ADDR_MULTI_SPEED_HEAD = 0x1200 + 0
ADDR_MULTI_SPEED_FIRST = 0x1200 + 20
ADDR_DI_FLAGS = 0x0300

ADDR_ANALOG_OFFSET = 0x0300 + 36
ADDR_ANALOG_TIME_CTE = 0x0300 + 37
ADDR_ANALOG_DEADZONE = 0x0300 + 39
ADDR_ANALOG_DRIFT = 0x0300 + 40
ADDR_ANALOG_SPEED_10V = 0x0300 + 51

ADDR_POS_SRC_SEL = 0x0500 + 0
ADDR_MULTI_POS_HEAD = 0x1100 + 0
ADDR_MULTI_POS_FIRST = 0x1100 + 12
ADDR_PULSE_PER_ROT = 0x0500 + 2

ADDR_VDI_FN1 = 0x1700 + 0
ADDR_VDI_FN2 = 0x1700 + 2

ADDR_POS_LOW_PASS = 0x0500 + 4
ADDR_POS_TIME_CTE = 0x0500 + 6

REG_P0B_07 = 0x0B07  # P0B-07 (abs pos, 32-bit) - historical naming
REG_P0B_09 = 0x0B09  # P0B-09 (pos per revolution, 0..65535)
REG_P0B_00 = 0x0B00  # P0B-00 (actual motor speed, rpm)
REG_P0B_33 = 0x0B21  # P0B-33 (fault selector/history index) BUG:WRONG ADDRESS
REG_P0B_34 = 0x0B22  # P0B-34 (fault code selected by P0B-33)BUG:WRONG ADDRESS
REG_P05_02 = 0x0502  # P05-02 (command units per revolution)
REG_P0C_26 = 0x0C26  # word order


@dataclass
class MotorConfig:
    port: str = "COM5"
    slave_id: int = 1
    baud_rate: int = 19200
    parity: str = "E"
    stop_bits: int = 1
    byte_size: int = 8

    @classmethod
    def from_config(cls, src: Mapping[str, Any]) -> MotorConfig:
        return cls(
            port=src["MOTOR_PORT"],
            slave_id=int(src["MOTOR_ID"]),
            baud_rate=int(src["MOTOR_BAUD_RATE"]),
            # parity/stop_bits/byte_size stay at defaults
        )

    def set_slave_id(self, slave_id):
        self.slave_id = slave_id
        return self

    def set_baudrate(self, baud_rate):
        self.baud_rate = baud_rate
        return self


class Motor:
    def __init__(self, config: MotorConfig | None):
        if not config:
            config = MotorConfig()  # default value
        try:
            self.client = ModbusSerialClient(
                port=config.port,
                baudrate=config.baud_rate,
                parity=config.parity,
                stopbits=config.stop_bits,
                bytesize=config.byte_size,
                timeout=(100 / 1_000_000) + 0.1,
            )
            self.client.connect()
            self.state = "READY"
            self.mode = "NONE"
        except Exception as e:
            raise RuntimeError(f"CONNECTION ERROR:::{e}")

    def __del__(self):
        self.client.close()

    def _write(self, addr: int, value: int):
        r = self.client.write_register(addr, value)
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS WRITE FAILED")

    def _write_u32(self, addr: int, value: int):
        assert np.abs(value) < (1 << 31), "POSITION TOO BIG"
        b_high = (value >> 16) & 0xFFFF
        b_low = value & 0xFFFF
        # print(b_high,b_low)
        r = self.client.write_registers(addr, [b_low, b_high])
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS WRITE FAILED")

    def _read(self, addr: int):
        r = self.client.read_holding_registers(addr)
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS READ FAILED")
        else:
            value = r.registers[0]
            return value

    def _read_many(self, addr: int, count: int):
        r = self.client.read_holding_registers(addr, count=count)
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS READ FAILED")
        else:
            return r.registers

    def start(self):
        print(f":::::STARTING MOTOR WITH MODE {self.mode}::::::")
        if self.mode == "NONE":
            raise RuntimeError("Trying to run motor on undefined mode.")
        self._write(ADDR_VDI_LEVEL, 0b11)
        self.state = "RUN"

    def stop(self):
        print(":::::STOPING MOTOR::::::")
        self._write(ADDR_VDI_LEVEL, 0)
        self.state = "STOP"

    def analog_mode(self):
        print(":::::SWITCHING TO ANALOG INPUT::::::")
        # NOTE:  I dont know what any of these do, keep it just to be safe.
        self._write(ADDR_VDI_LEVEL, 0)
        self._write(ADDR_VDI_ENABLE, 1)
        self._write(ADDR_DI1_FN, 0)

        # Set mode to speed
        self._write(ADDR_CTRL_MODE, 0)

        # sets speed command to analog A
        data = [1, 2, 0]
        self.client.write_registers(ADDR_SPEED_SRC_A, data)

        # set Analog input params
        self._write(ADDR_ANALOG_OFFSET, 0)

        self._write(ADDR_ANALOG_TIME_CTE, 0)
        self._write(ADDR_ANALOG_DEADZONE, 0)
        self._write(ADDR_ANALOG_DRIFT, 0)
        self._write(ADDR_ANALOG_DRIFT, 0)
        self._write(ADDR_ANALOG_SPEED_10V, 500)
        self.mode = "VEL_ANALOG"

    def pos_ctrl(self, f, times):
        assert len(times) <= 16, "TIMES ARRAY CANT BE BIGGER THAN 16 ELEMENTS"
        print(":::::SETTING POSITION SINE::::::")
        PULSE_PER_ROT = 10000
        # NOTE:  I dont know what any of these do, keep it just to be safe.
        self._write(ADDR_VDI_LEVEL, 0)
        self._write(ADDR_VDI_ENABLE, 1)
        self._write(ADDR_DI1_FN, 0)

        self._write(ADDR_VDI_FN1, 1)
        self._write(ADDR_VDI_FN2, 28)

        # Set mode to position
        self.client.write_register(ADDR_CTRL_MODE, 1)

        self.client.write_register(ADDR_POS_SRC_SEL, 2)  # muti segment
        self.client.write_register(ADDR_PULSE_PER_ROT, 500)  # PULSE_PER_ROT )
        self.client.write_register(ADDR_POS_LOW_PASS, 10)
        self.client.write_register(ADDR_POS_TIME_CTE, 10)

        n_segments = len(times)

        TIME_UNIT = 0
        data = [1, n_segments, 0, TIME_UNIT, 0, 1]  # header
        self.client.write_registers(ADDR_MULTI_POS_HEAD, data)

        data = []
        t = 0
        x0 = f(times[-1])
        addr = ADDR_MULTI_POS_FIRST
        for dt in times:
            t = t + dt
            raw = f(t)  # voltas
            x1 = int(np.floor(PULSE_PER_ROT * raw))
            self._write_u32(addr, x1)
            addr = addr + 2
            speed = int((x1 - x0) / dt) & 0xFFFF
            print(x0, x1, speed)
            x0 = x1
            self._write(addr, speed)
            addr = addr + 1
            accel = 0  # int(500*dt)#TODO:DO CALC ON THIS
            self._write(addr, accel)
            addr = addr + 1
            wait = 0
            self._write(addr, wait)
            addr = addr + 1

        read = self._read_many(ADDR_MULTI_POS_FIRST, 5 * len(times))
        # print(read)
        # for a,b in zip(data,read):
        # assert(a==b),f"VALUE WRITE ERROR: TRIED TO WRITE {a}, READ BACK IS {b}"
        self.mode = "POS_SEC"

    # NOTE: THIS THROWS ERROR
    def vel_ctrl(self, f, times):
        assert len(times) <= 16, "TIMES ARRAY CANT BE BIGGER THAN 16 ELEMENTS"
        print(":::::SETTING SPEED SINE::::::")
        self._write(ADDR_VDI_LEVEL, 0)
        self._write(ADDR_VDI_ENABLE, 1)
        self._write(ADDR_DI1_FN, 0)

        # Set mode to speed
        self._write(ADDR_CTRL_MODE, 0)

        # sets speed command to multi segment
        data = [0, 5, 1]
        self.client.write_registers(ADDR_SPEED_SRC_A, data)
        # set multi speed params

        n_segments = len(times)

        data = [1, n_segments, 0]  # header
        self.client.write_registers(ADDR_MULTI_SPEED_HEAD, data)

        data = []
        t = 0
        for dt in times:
            t = t + dt
            speed = int(np.floor(f(t)))
            if speed < 0:
                speed = speed & 0xFFFF
            print(speed)
            data.append(speed)
            data.append(int(dt))
            data.append(0)

        # self._write_many(ADDR_MULTI_SPEED_FIRST,data)
        for i, d in enumerate(data):
            self._write(ADDR_MULTI_SPEED_FIRST + i, d)
            v = self._read(ADDR_MULTI_SPEED_FIRST + i)
            assert d == v, f"VALUE WRITE ERROR: TRIED TO WRITE {d}, READ BACK IS {v}"
        # self.client.write_registers(ADDR_MULTI_SPEED_FIRST,data)
        self.mode = "VEL_SEC"


if __name__ == "__main__":
    raise RuntimeError("Motor is a library, it cannot be run directly")
