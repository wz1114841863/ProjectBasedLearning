#! /usr/bin/env python
import sys
import random
from optparse import OptionParser

# 模拟程序运行时进程状态如何变化以及使用CPU或IO操作
# process switch behavior
SCHED_SWITCH_ON_IO = 'SWITCH_ON_IO'
SCHED_SWITCH_ON_END = 'SWITCH_ON_END'

# io finished behavior
IO_RUN_LATER = 'IO_RUN_LATER'
IO_RUN_IMMEDIATE = 'IO_RUN_IMMEDIATE'

# process states
STATE_RUNNING = 'RUNNING'
STATE_READY = 'READY'
STATE_DONE = 'DONE'
STATE_WAIT = 'BLOCKED'

# members of process structure
PROC_CODE = 'code_'
PROC_PC = 'pc_'
PROC_ID = 'pid_'
PROC_STATE = 'proc_state_'

# things a process can do
DO_COMPUTE = 'cpu'
DO_IO = 'io'
DO_IO_DONE = 'io_done'


def random_seed(seed):
    try:
        random.seed(seed, version=1)
    except:
        random.seed(seed)
    return


def get_parser():
    """ 命令行参数解析 """
    # optparse 在 Python 3.2 后被弃用,建议使用更强大的 argparse 模块.
    parser = OptionParser()
    parser.add_option('-s', '--seed', default=0, help='the random seed',
                      action='store', type='int', dest='seed')
    parser.add_option('-P', '--program', default='', help='more specific controls over programs',
                      action='store', type='string', dest='program')
    parser.add_option('-l', '--processlist', default='', help='a comma-separated list of processes to run, in the form X1:Y1,X2:Y2,... where X is the number of instructions that process should run, and Y the chances (from 0 to 100) that an instruction will use the CPU or issue an IO (i.e., if Y is 100, a process will ONLY use the CPU and issue no I/Os; if Y is 0, a process will only issue I/Os)', action='store', type='string', dest='process_list')
    parser.add_option('-L', '--iolength', default=5, help='how long an IO takes',
                      action='store', type='int', dest='io_length')
    parser.add_option('-S', '--switch', default='SWITCH_ON_IO', help='when to switch between processes: SWITCH_ON_IO, SWITCH_ON_END',
                      action='store', type='string', dest='process_switch_behavior')
    parser.add_option('-I', '--iodone', default='IO_RUN_LATER', help='type of behavior when IO ends: IO_RUN_LATER, IO_RUN_IMMEDIATE',
                      action='store', type='string', dest='io_done_behavior')
    parser.add_option('-c', help='compute answers for me',
                      action='store_true', default=False, dest='solve')
    parser.add_option('-p', '--printstats', help='print statistics at end; only useful with -c flag (otherwise stats are not printed)',
                      action='store_true', default=False, dest='print_stats')
    options, _ = parser.parse_args()
    assert (options.process_switch_behavior ==
            SCHED_SWITCH_ON_IO or options.process_switch_behavior == SCHED_SWITCH_ON_END)
    assert (options.io_done_behavior ==
            IO_RUN_IMMEDIATE or options.io_done_behavior == IO_RUN_LATER)

    return options


class scheduler:
    def __init__(self, process_switch_behavior, io_done_behavior, io_length):
        self.proc_info = {}  # 进程列表
        self.process_switch_behavior = process_switch_behavior  # 什么时候切换进程
        self.io_done_behavior = io_done_behavior  # IO操作结束时的行为
        self.io_length = io_length  # IO操作执行需要的时间
        return

    def new_process(self):
        """ 创建一个新进程, 进程ID自动加一 """
        proc_id = len(self.proc_info)
        self.proc_info[proc_id] = {}
        self.proc_info[proc_id][PROC_PC] = 0
        self.proc_info[proc_id][PROC_ID] = proc_id
        self.proc_info[proc_id][PROC_CODE] = []
        self.proc_info[proc_id][PROC_STATE] = STATE_READY
        return proc_id

    def load_program(self, program):
        """ 加载程序
        程序格式如下: c7,i,c1,i
        解释: 计算7次, 接着进入IO操作, 接着计算一次, 接着进入IO操作
        """
        proc_id = self.new_process()
        for line in program.split(','):
            op_code = line[0]
            if op_code == 'c':
                num = int(line[1:])
                for i in range(num):
                    self.proc_info[proc_id][PROC_CODE].append(DO_COMPUTE)
            elif op_code == 'i':
                self.proc_info[proc_id][PROC_CODE].append(DO_IO)
                # add one compute to HANDLE the I/O completion
                self.proc_info[proc_id][PROC_CODE].append(DO_IO_DONE)
            else:
                print('bad opcode %s (should be c or i)' % op_code)
                exit(1)
        return

    def load(self, program_description):
        """ 加载程序
        程序格式如下: x:y
        解释: x代表指令个数, y代表当前指令是CPU操作还是IO操作的比例.
        """
        proc_id = self.new_process()
        tmp = program_description.split(':')
        if len(tmp) != 2:
            print('Bad description (%s): Must be number <x:y>' %
                  program_description)
            print('  where X is the number of instructions')
            print('  and Y is the percent change that an instruction is CPU not IO')
            exit(1)

        num_instructions, chance_cpu = int(tmp[0]), float(tmp[1]) / 100.0
        for i in range(num_instructions):
            if random.random() < chance_cpu:
                self.proc_info[proc_id][PROC_CODE].append(DO_COMPUTE)
            else:
                self.proc_info[proc_id][PROC_CODE].append(DO_IO)
                # add one compute to HANDLE the I/O completion
                self.proc_info[proc_id][PROC_CODE].append(DO_IO_DONE)
        return

    def move_to_ready(self, expected, pid=-1):
        if pid == -1:
            pid = self.curr_proc
        assert (self.proc_info[pid][PROC_STATE] == expected)
        self.proc_info[pid][PROC_STATE] = STATE_READY
        return

    def move_to_wait(self, expected):
        assert (self.proc_info[self.curr_proc][PROC_STATE] == expected)
        self.proc_info[self.curr_proc][PROC_STATE] = STATE_WAIT
        return

    def move_to_running(self, expected):
        assert (self.proc_info[self.curr_proc][PROC_STATE] == expected)
        self.proc_info[self.curr_proc][PROC_STATE] = STATE_RUNNING
        return

    def move_to_done(self, expected):
        assert (self.proc_info[self.curr_proc][PROC_STATE] == expected)
        self.proc_info[self.curr_proc][PROC_STATE] = STATE_DONE
        return

    def next_proc(self, pid=-1):
        """ 进程切换, pid指定时切换到指定进程, 否则顺序查看接下来的进程是否处于STATE_READY """
        if pid != -1:
            # 根据进程ID切换对应进程至运行态
            self.curr_proc = pid
            self.move_to_running(STATE_READY)
            return
        for pid in range(self.curr_proc + 1, len(self.proc_info)):
            if self.proc_info[pid][PROC_STATE] == STATE_READY:
                self.curr_proc = pid
                self.move_to_running(STATE_READY)
                return
        for pid in range(0, self.curr_proc + 1):
            if self.proc_info[pid][PROC_STATE] == STATE_READY:
                self.curr_proc = pid
                self.move_to_running(STATE_READY)
                return
        return

    def get_num_processes(self):
        return len(self.proc_info)

    def get_num_instructions(self, pid):
        return len(self.proc_info[pid][PROC_CODE])

    def get_instruction(self, pid, index):
        return self.proc_info[pid][PROC_CODE][index]

    def get_num_active(self):
        """ 获取活跃进程数 """
        num_activate = 0
        for pid in range(len(self.proc_info)):
            if self.proc_info[pid][PROC_STATE] != STATE_DONE:
                num_activate += 1
        return num_activate

    def get_num_runnable(self):
        """ 获取可运行进程数 """
        num_active = 0
        for pid in range(len(self.proc_info)):
            if self.proc_info[pid][PROC_STATE] == STATE_READY or \
                    self.proc_info[pid][PROC_STATE] == STATE_RUNNING:
                num_active += 1
        return num_active

    def get_ios_in_flight(self, current_time):
        """ 统计每个流程的未完成IO """
        num_in_flight = 0
        for pid in range(len(self.proc_info)):
            for t in self.io_finish_times[pid]:
                if t > current_time:
                    num_in_flight += 1
        return num_in_flight

    def check_for_switch(self):
        return

    def space(self, num_columns):
        for i in range(num_columns):
            print('%10s' % ' ', end='')

    def check_if_done(self):
        """ 退出执行完毕的进程 """
        if len(self.proc_info[self.curr_proc][PROC_CODE]) == 0:
            if self.proc_info[self.curr_proc][PROC_STATE] == STATE_RUNNING:
                self.move_to_done(STATE_RUNNING)
                self.next_proc()
        return

    def run(self):
        clock_tick = 0
        if len(self.proc_info) == 0:
            return

        self.io_finish_times = {}
        for pid in range(len(self.proc_info)):
            self.io_finish_times[pid] = []

        # make first ont active
        self.curr_proc = 0
        self.move_to_running(STATE_READY)

        # OUTPUT: header fro each column
        print('%s' % 'Time', end='')
        for pid in range(len(self.proc_info)):
            print('%14s' % ('PID:%2d' % (pid)), end='')
        print('%14s' % 'CPU', end='')
        print('%14s' % 'IOs', end='')
        print('')

        # init statistics
        io_busy = 0
        cpu_busy = 0

        while self.get_num_active() > 0:
            clock_tick += 1
            io_done = False
            for pid in range(len(self.proc_info)):
                if clock_tick in self.io_finish_times[pid]:
                    io_done = True
                    self.move_to_ready(STATE_WAIT, pid)
                    if self.io_done_behavior == IO_RUN_IMMEDIATE:
                        # IO_RUN_IMMEDIATE
                        if self.curr_proc != pid:
                            if self.proc_info[self.curr_proc][PROC_STATE] == STATE_RUNNING:
                                self.move_to_ready(STATE_RUNNING)
                        self.next_proc(pid)
                    else:
                        # IO_RUN_LATER
                        if self.process_switch_behavior == SCHED_SWITCH_ON_END and self.get_num_runnable() > 1:
                            # this means the process that issued the io should be run
                            self.next_proc(pid)
                        if self.get_num_runnable() == 1:
                            # this is the only thing to run: so run it
                            self.next_proc(pid)
                    self.check_if_done()

            # if current proc is RUNNING and has an instruction, execute it
            instruction_to_execute = ''
            if self.proc_info[self.curr_proc][PROC_STATE] == STATE_RUNNING and \
                    len(self.proc_info[self.curr_proc][PROC_CODE]) > 0:
                instruction_to_execute = self.proc_info[self.curr_proc][PROC_CODE].pop(
                    0)
                cpu_busy += 1

            # OUTPUT: print what everyone is up to
            if io_done:
                print('%3d*' % clock_tick, end='')
            else:
                print('%3d ' % clock_tick, end='')
            for pid in range(len(self.proc_info)):
                if pid == self.curr_proc and instruction_to_execute != '':
                    print('%14s' % ('RUN:'+instruction_to_execute), end='')
                else:
                    print('%14s' % (self.proc_info[pid][PROC_STATE]), end='')

            # CPU output here: if no instruction executes, output a space, otherwise a 1
            if instruction_to_execute == '':
                print('%14s' % ' ', end='')
            else:
                print('%14s' % '1', end='')

            # IO output here:
            num_outstanding = self.get_ios_in_flight(clock_tick)
            if num_outstanding > 0:
                print('%14s' % str(num_outstanding), end='')
                io_busy += 1
            else:
                print('%10s' % ' ', end='')
            print('')

            # if this is an IO start instruction, switch to waiting state
            # and add an io completion in the future
            if instruction_to_execute == DO_IO:
                self.move_to_wait(STATE_RUNNING)
                self.io_finish_times[self.curr_proc].append(
                    clock_tick + self.io_length + 1)
                if self.process_switch_behavior == SCHED_SWITCH_ON_IO:
                    self.next_proc()

            # ENDCASE: check if currently running thing is out of instructions
            self.check_if_done()
        return (cpu_busy, io_busy, clock_tick)


if __name__ == "__main__":
    options = get_parser()
    random_seed(options.seed)
    s = scheduler(options.process_switch_behavior,
                  options.io_done_behavior, options.io_length)
    if options.program != '':
        for p in options.program.split(':'):
            s.load_program(p)
    else:
        for p in options.process_list.split(','):
            s.load(p)

    assert (options.io_length >= 0)

    if options.solve == False:
        print('Produce a trace of what would happen when you run these processes:')
        for pid in range(s.get_num_processes()):
            print(f'Process {pid}')
            for inst in range(s.get_num_instructions(pid)):
                print('  %s' % s.get_instruction(pid, inst))
            print('')
        print('Important behaviors:')
        print('  System will switch when ', end='')
        if options.process_switch_behavior == SCHED_SWITCH_ON_IO:
            print('the current process is FINISHED or ISSUES AN IO')
        else:
            print('the current process is FINISHED')
        print('  After IOs, the process issuing the IO will ', end='')
        if options.io_done_behavior == IO_RUN_IMMEDIATE:
            print('run IMMEDIATELY')
        else:
            print('run LATER (when it is its turn)')
        print('')
        exit(0)

    cpu_busy, io_busy, clock_tick = s.run()
    if options.print_stats:
        print('')
        print('Stats: Total Time %d' % clock_tick)
        print('Stats: CPU Busy %d (%.2f%%)' %
              (cpu_busy, 100.0 * float(cpu_busy) / clock_tick))
        print('Stats: IO Busy  %d (%.2f%%)' %
              (io_busy, 100.0 * float(io_busy) / clock_tick))
        print('')
