from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
import numpy as np

CMD_RESP_US = 50000
CMD_BYTE_US = 50000
FAST_RESP_US = 3000
FAST_BYTE_US = 2000
RAMP_TIME_S = 3.0
DRIVE_PREFLIGHT_READS = 3
DRIVE_PREFLIGHT_ATTEMPTS = 30
ENCODER_CAL_POSITION_LOSS_S = 2.0
RECIP_ENCODER_REVERSE_DELAY_S = 0.020
RECIP_ENCODER_ORIGIN_TIMEOUT_S = 3.0

def make_port_path(short_name: str) -> str:
    """Convert COM port name to Windows device path."""
    if short_name.upper().startswith('COM'):
        num = int(short_name[3:]) if short_name[3:].isdigit() else 0
        if num >= 10:
            return f"\\\\.\\{short_name}"
    return short_name


   
#Modbus client params
MODBUS_PORT=f"COM{1}"
MODBUS_SLAVE_ID=1
MODBUS_BAUD_RATE=115200
MODBUS_PARITY="E"
MODBUS_STOPBITS=1
MODBUS_BYTESIZE=8

#Command values
START_CMD = 0x0001
STOP_CMD = 0x0000


# Controller memory addresses
ADDR_SPEED_VAL = 0x0603  # 1540-1 : RPM setpoint (int16)
ADDR_CTRL_MODE = 0x0200  # control mode (speed/pos/torque)
ADDR_DI1_FN = 0x0302  # DI1 func (disable)
ADDR_SPEED_SRC_A = 0x0600  # main speed source A
ADDR_SPEED_SRC_B = 0x0601  # aux speed source B
ADDR_SPEED_SRC_SEL = 0x0602  # speed command source
ADDR_VDI_ENABLE = 0x0C09  # comm VDI enable
ADDR_VDI_LEVEL = 0x3100  # VDI virtual
ADDR_MULTI_SPEED_HEAD=0x1200
ADDR_MULTI_SPEED_FIRST=0x1220
ADDR_DI_FLAGS=0x0300

ADDR_ANALOG_OFFSET=0x0336
ADDR_ANALOG_TIME_CTE=0x0337
ADDR_ANALOG_DEADZONE=0x0339
ADDR_ANALOG_DRIFT=0x0340
ADDR_ANALOG_SPEED_10V=0x0351

ADDR_POS_SRC_SEL=0x0500
ADDR_MULTI_POS_HEAD=0x1100
ADDR_MULTI_POS_FIRST=0x1112

REG_P0B_07 = 0x0B07  # P0B-07 (abs pos, 32-bit) - historical naming
REG_P0B_09 = 0x0B09  # P0B-09 (pos per revolution, 0..65535)
REG_P0B_00 = 0x0B00  # P0B-00 (actual motor speed, rpm)
REG_P0B_33 = 0x0B21  # P0B-33 (fault selector/history index) BUG:WRONG ADDRESS
REG_P0B_34 = 0x0B22  # P0B-34 (fault code selected by P0B-33)BUG:WRONG ADDRESS
REG_P05_02 = 0x0502  # P05-02 (command units per revolution)
REG_P0C_26 = 0x0C26  # word order


    
def motor_start(motor): 
    print(":::::STARTING MOTOR::::::")
    motor.write_register(ADDR_VDI_LEVEL, 1 )

def motor_stop(motor):
    print(":::::STOPING MOTOR::::::")
    motor.write_register(ADDR_VDI_LEVEL, 0 )


def motor_speed_mode(motor):
    SLAVE_ID=1
    # P31-00 = 0 (VDI STOP)
    motor.write_register(ADDR_VDI_LEVEL, 0 )    
    # P0C-09 = 1 (Comm VDI)
    motor.write_register(ADDR_VDI_ENABLE, 1)
       
    # P03-02 = 0 (DI1 func)
    motor.write_register(ADDR_DI1_FN, 0)
        
    # P02-00 = 0 (control mode speed)
    motor.write_register(ADDR_CTRL_MODE, 0)
        
    # P06-00 = 0 (A src = given)
    motor.write_register(ADDR_SPEED_SRC_A, 0)
        
    # P06-01 = 3 (B src = 0)
    motor.write_register(ADDR_SPEED_SRC_B, 3)
        
    # P06-02 = 0 (sel = A)
    motor.write_register(ADDR_SPEED_SRC_SEL, 0)


#NOTE: THIS THROWS ERROR
def motor_pos_sin(motor): 
    print(":::::SETTING SPEED SINE::::::")
    #NOTE:  I dont know what any of these do, keep it just to be safe.
    motor.write_register(ADDR_VDI_LEVEL, 0 )  
    motor.write_register(ADDR_VDI_ENABLE, 1 ) 
    motor.write_register(ADDR_DI1_FN, 0 )
    
    # Set mode to position
    motor.write_register(ADDR_CTRL_MODE, 1 )
       
    motor.write_register(ADDR_POS_SRC_SEL,2) #muti segment


    #sets speed command to multi segment
    data=[0,5,1]
    #set multi speed params

    SEGMENT_TIME=1 # 0.1 secs
    NUM_SEGMENTS=2
    MAX_VALUE=120

    data=[1,NUM_SEGMENTS,1,1,0,0]  #header  
    motor.write_registers(ADDR_MULTI_POS_HEAD,data)

    data=[]
    points = np.linspace(0, 2 * np.pi, NUM_SEGMENTS)
    for i in points:
        pos=np.floor(MAX_VALUE * np.sin(i))
        pos_high=(pos >> 16)& 0xFFFF
        pos_low=pos & 0xFFFF
        #if pos<0:
        #    pos_high=pos >> 16
        #    pos_low=pos & 0xFFFF
        data.append(pos_high)
        data.append(pos_low)
        speed=100#TODO:do calc on this
        data.append(speed)
        accel=100#TODO:DO CALC ON THIS
        data.append(accel)
        wait=0
        data.append(wait)
        data.append(SEGMENT_TIME)
        data.append(0)

    motor.write_registers(ADDR_MULTI_POS_FIRST,data)
    read=motor.read_holding_registers(ADDR_MULTI_POS_FIRST,len(data))
    print(read)



#NOTE: THIS THROWS ERROR
def motor_vel_sin(motor): 
    print(":::::SETTING SPEED SINE::::::")
    #NOTE:  I dont know what any of these do, keep it just to be safe.
    motor.write_register(ADDR_VDI_LEVEL, 0 )  
    motor.write_register(ADDR_VDI_ENABLE, 1 ) 
    motor.write_register(ADDR_DI1_FN, 0 )
    
    # Set mode to speed
    motor.write_register(ADDR_CTRL_MODE, 0 )
        

    #sets speed command to multi segment
    data=[0,5,1]
    motor.write_registers(ADDR_SPEED_SRC_A,data)
    #set multi speed params

    SEGMENT_TIME=1 # 0.1 secs
    NUM_SEGMENTS=2
    MAX_VALUE=120

    data=[1,NUM_SEGMENTS,0]  #header 
    motor.write_registers(ADDR_MULTI_SPEED_HEAD,data)
    

    data=[]
    points = np.linspace(0, 2 * np.pi, NUM_SEGMENTS)
    for i in points:
        speed=np.floor(MAX_VALUE * np.sin(i))
        if speed<0:
            speed=speed & 0xFFFF
        data.append(speed)
        data.append(SEGMENT_TIME)
        data.append(0)

    motor.write_registers(ADDR_MULTI_SPEED_FIRST,data)
    read=motor.read_holding_registers(ADDR_MULTI_SPEED_FIRST)
    print(read)


#NOTE: THIS THROWS ERROR
def motor_switch_analog(motor): 
    print(":::::SWITCHING TO ANALOG INPUT::::::")
    #NOTE:  I dont know what any of these do, keep it just to be safe.
    motor.write_register(ADDR_VDI_LEVEL, 0 )  
    motor.write_register(ADDR_VDI_ENABLE, 1 ) 
    motor.write_register(ADDR_DI1_FN, 0 )
    
    # Set mode to speed
    motor.write_register(ADDR_CTRL_MODE, 0 )
        

    #sets speed command to analog A
    data=[1,2,0]
    motor.write_registers(ADDR_SPEED_SRC_A,data)
    
    #set Analog input params 
    motor.write_registers(ADDR_ANALOG_OFFSET,0)

    motor.write_register(ADDR_ANALOG_TIME_CTE,0)
    motor.write_register(ADDR_ANALOG_DEADZONE,0)
    motor.write_register(ADDR_ANALOG_DRIFT,0)
    motor.write_register(ADDR_ANALOG_DRIFT,0)
    motor.write_register(ADDR_ANALOG_SPEED_10V,500)
    
    #read=motor.read_holding_registers(ADDR_MULTI_SPEED_FIRST)
    #print(read)


def try_loop(func):
    N_TRIES=5
    while N_TRIES:
        try:
            return func()
        except Exception as e:
            print(f"THREW Exception {e}, retrying operation")
            N_TRIES=N_TRIES-1

#NOTE:  MAIN SCRIPT 


try:
    client = ModbusSerialClient(
               port=MODBUS_PORT,
                baudrate=MODBUS_BAUD_RATE,
                parity=MODBUS_PARITY,
                stopbits=MODBUS_STOPBITS,
                bytesize=MODBUS_BYTESIZE,
                timeout=(100 / 1_000_000) + 0.1
            )
    client.connect()
except Exception as e:
    print(f"CONNECTION ERROR:::{e}")
    exit(1)            

#try_loop(lambda:motor_stop(client))

try_loop(lambda:motor_set_sine(client))
try_loop(lambda:motor_start(client))

#closes modbus connection
client.close()
   
