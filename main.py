from motor import Motor
import dlg400 as dlg

#NOTE:  MAIN SCRIPT 
def try_loop(func):
    N_TRIES=5
    while N_TRIES:
        try:
            return func()
        except Exception as e:
            print(f"THREW Exception {e}, retrying operation")
            N_TRIES=N_TRIES-1

#NOTE: DLG SET UP
DLG_ARGS=None
cfg = dlg.parse_args(DLG_ARGS)
if cfg is None:
    dlg.print_usage()
    exit()
dlg.run(cfg)


#NOTE: MOTOR SETUP

NUM_SEGMENTS=16
SEG_TIME=1
times = np.ones(NUM_SEGMENTS)*SEG_TIME 
period=np.sum(times)
motor=Motor()    

angular_speed=2*np.pi/(period)
print("freq:",angular_speed)
MAX_VAL=50

def test(i):
    freq=16
    v=math.fmod(i,angular_speed)/angular_speed
    if v>=0.5:
        return -MAX_VAL
    else:
        return MAX_VAL

f_sine=lambda i: 100*MAX_VAL*np.sin(angular_speed*i)

r_sine=lambda i: 0.1*np.sin(angular_speed*i)

try_loop(lambda:motor.stop())

#motor.vel_ctrl(test,times)
#motor.vel_ctrl(f_sine,times)
motor.pos_ctrl(r_sine,times)

try_loop(lambda:motor.start())
