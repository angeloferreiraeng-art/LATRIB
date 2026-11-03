# modbus_virtual_slave.py
from pymodbus.server import StartTcpServer, StartSerialServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
import logging

# Enable logging to see detailed communication
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

def run_virtual_slave(mode="tcp"):
    """Start a virtual Modbus slave for testing."""
    # 1. Create data blocks with some test data
    # Parameters: start_address, [initial_values]
    # Holding Registers (4x): commonly used for read/write values
    holding_registers = ModbusSequentialDataBlock(0, [10, 20, 30, 40, 50])
    # Coils (0x): used for boolean/on-off states
    coils = ModbusSequentialDataBlock(0, [True, False, True, False])
    # Input Registers (3x): read-only values
    input_registers = ModbusSequentialDataBlock(0, [100, 200, 300])
    # Discrete Inputs (1x): read-only boolean values
    discrete_inputs = ModbusSequentialDataBlock(0, [1, 0, 1])

    # 2. Create slave context (slave ID 1 is standard)
    slave_context = ModbusSlaveContext(
        hr=holding_registers,
        co=coils,
        ir=input_registers,
        di=discrete_inputs
    )

    # 3. Create server context (single = True means only one slave)
    context = ModbusServerContext(slaves={1: slave_context}, single=True)

    print(f"Starting virtual Modbus {mode.upper()} server on slave ID 1...")

    # 4. Start the server based on the chosen mode
    if mode == "tcp":
        # For TCP: client connects to localhost on port 5020
        StartTcpServer(context, address=("127.0.0.1", 5020))
    elif mode == "rtu":
        # For RTU: you need a virtual serial port pair or USB converter
        # On Windows, use com0com to create virtual ports like COM3 and COM4
        # On Linux, use `socat` to create virtual ports
        settings = {
            "port": "COM3",   # Change to your virtual port
            "baudrate": 19200,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1
        }
        StartSerialServer(context, **settings)
    else:
        print("Mode must be 'tcp' or 'rtu'")

if __name__ == "__main__":
    # Run TCP server by default, change to "rtu" if needed
    run_virtual_slave("tcp")