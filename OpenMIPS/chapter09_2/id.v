`timescale 1ns / 1ps
`include "defines.v"

/*
    对指令进行译码,得到最终运算的相关信息
*/

module id(
    input wire rst,

    // 来自译码阶段的指令和其对应的地址
    input wire [`InstAddrBus] pc_i,
    input wire [`InstBus] inst_i,

    // Regfile返回的信息
    input wire [`RegBus] reg1_data_i,  // 第一个读寄存器的输入
    input wire [`RegBus] reg2_data_i,  // 第二个读寄存器的输入

    // 处于执行阶段的指令的运算结果
    input wire ex_wreg_i,
    input wire [`RegBus] ex_wdata_i,
    input wire [`RegAddrBus] ex_wd_i,

    // 处于访存阶段的指令的运算结果
    input wire mem_wreg_i,
    input wire [`RegBus] mem_wdata_i,
    input wire [`RegAddrBus] mem_wd_i,

    // 当前处于译码阶段的指令是否位于延迟槽
    input wire is_in_delayslot_i,

    //
    input wire [`AluOpBus] ex_aluop_i,

    // 新增加的输出接口
    output wire [`RegBus] inst_o,

    // 输出到Regfile的信息
    output reg reg1_read_o,  // 第一个读寄存器的读使能信号
    output reg reg2_read_o,  // 第二个读寄存器的读使能信号
    output reg [`RegAddrBus] reg1_addr_o,  // 第一个读寄存器的读地址信号
    output reg [`RegAddrBus] reg2_addr_o,  // 第二个读寄存器的读地址信号

    // 送到执行阶段的信息
    output reg [`AluOpBus] aluop_o,  // 运算的子类型
    output reg [`AluSelBus] alusel_o,  // 运算的类型
    output reg [`RegBus] reg1_o,  // 源操作数1
    output reg [`RegBus] reg2_o,  // 源操作数2
    output reg [`RegAddrBus] wd_o,  // 要写入的目的寄存器地址
    output reg wreg_o,  // 是否有要写入的目的寄存器

    // 流水线暂停请求信号
    output wire stallreq,

    // 转移指令相关
    output reg next_inst_in_delayslot_o,

    output reg branch_flag_o,
    output reg [`RegBus] branch_target_address_o,
    output reg [`RegBus] link_addr_o,
    output reg is_in_delayslot_o
);

    // 取得指令的指令码, 功能码
    // 对于ori的指令只需通过判断第26 - 31bit的值,既可判断是否是ori指令
    wire [5: 0] op = inst_i[31: 26];  // 指令码
    wire [4: 0] op2 = inst_i[10: 6];  // 用于判断SPECIAL
    wire [5: 0] op3 = inst_i[5: 0];  // 功能码
    wire [4: 0] op4 = inst_i[20: 16];  // 用于判断SPECIAL

    // 保存指令执行需要的立即数
    reg [`RegBus] imm;

    // 指示指令是否有效
    reg instvalid;

    // 转移指令
    wire [`RegBus] pc_plus_8;
    wire [`RegBus] pc_plus_4;

    wire [`RegBus] imm_sll2_signedext;  // 左移两位,符号扩展

    assign pc_plus_8 = pc_i + 8;  // 保存当前译码阶段指令后面第二条指令的地址
    assign pc_plus_4 = pc_i + 4;  // 保存当前译码阶段指令后面紧接着的指令的地址
    assign imm_sll2_signedext = {{14{inst_i[15]}}, inst_i[15: 0], 2'b00};

    // inst_o的值就是译码阶段的指令
    assign inst_o = inst_i;

    // 表示要读取的寄存器1是否与上一条指令存在load相关
    reg stallreq_for_reg1_loadrelate;
    // 表示要读取的寄存器2是否与上一条指令存在load相关
    reg stallreq_for_reg2_loadrelate;
    // 新定义的一个变量, 表示上一条指令是否是加载指令
    wire pre_inst_is_load;

    // 依据输入信号ex_aluop_i的值, 判断上一条指令是否是加载指令
    // 如果是加载指令, 那么置pre_inst_is_load为1, 反之置为0
    assign pre_inst_is_load = ((ex_aluop_i == `EXE_LB_OP)  ||
                               (ex_aluop_i == `EXE_LBU_OP) ||
                               (ex_aluop_i == `EXE_LH_OP)  ||
                               (ex_aluop_i == `EXE_LHU_OP) ||
                               (ex_aluop_i == `EXE_LW_OP)  ||
                               (ex_aluop_i == `EXE_LWR_OP) ||
                               (ex_aluop_i == `EXE_LWL_OP) ||
                               (ex_aluop_i == `EXE_LL_OP)  ||
                               (ex_aluop_i == `EXE_SC_OP)) ? 1'b1 : 1'b0;

    // 对指令进行译码
    always @(*) begin
        if (rst == `RstEnable) begin
            aluop_o <= `EXE_NOP_OP;
            alusel_o <= `EXE_RES_NOP;
            wd_o <= `NOPRegAddr;
            wreg_o <= `WriteDisable;
            instvalid <= `InstValid;
            reg1_read_o <= 1'b0;
            reg2_read_o <= 1'b0;
            reg1_addr_o <= `NOPRegAddr;
            reg2_addr_o <= `NOPRegAddr;
            imm <= 32'h0;
        end else begin
            aluop_o <= `EXE_NOP_OP;
            alusel_o <= `EXE_RES_NOP;
            wd_o <= inst_i[15: 11];  // 默认目的寄存器地址wd_o
            wreg_o <= `WriteDisable;
            instvalid <= `InstInvalid;
            reg1_read_o <= 1'b0;
            reg2_read_o <= 1'b0;
            reg1_addr_o <= inst_i[25: 21];  // 默认通过Regfile读端口1读取的寄存器地址
            reg2_addr_o <= inst_i[20: 16];  // 默认通过Regfile读端口2读取的寄存器地址
            imm <= `ZeroWord;

            case (op)
                `EXE_SPECIAL_INST: begin  // 指令时SPECIAL
                    case (op2)
                        5'b00000: begin  // 必须是5'b00000
                            case (op3)
                                `EXE_OR: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_OR_OP;
                                    alusel_o <= `EXE_RES_LOGIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_AND: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_AND_OP;
                                    alusel_o <= `EXE_RES_LOGIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_XOR: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_XOR_OP;
                                    alusel_o <= `EXE_RES_LOGIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_NOR: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_NOR_OP;
                                    alusel_o <= `EXE_RES_LOGIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_SLLV: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_SLL_OP;
                                    alusel_o <= `EXE_RES_SHIFT;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_SRLV: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_SRL_OP;
                                    alusel_o <= `EXE_RES_SHIFT;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_SRAV: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_SRA_OP;
                                    alusel_o <= `EXE_RES_SHIFT;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_SYNC: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_NOP_OP;
                                    alusel_o <= `EXE_RES_NOP;
                                    reg1_read_o <= 1'b0;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_MFHI: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_MFHI_OP;
                                    alusel_o <= `EXE_RES_NOP;
                                    reg1_read_o <= 1'b0;
                                    reg2_read_o <= 1'b0;
                                    instvalid <= `InstValid;
                                end
                                `EXE_MFLO: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_MFLO_OP;
                                    alusel_o <= `EXE_RES_NOP;
                                    reg1_read_o <= 1'b0;
                                    reg2_read_o <= 1'b0;
                                    instvalid <= `InstValid;
                                end
                                `EXE_MTHI: begin
                                    wreg_o <= `WriteDisable;
                                    aluop_o <= `EXE_MTHI_OP;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b0;
                                    instvalid <= `InstValid;
                                end
                                `EXE_MTLO: begin
                                    wreg_o <= `WriteDisable;
                                    aluop_o <= `EXE_MTLO_OP;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b0;
                                    instvalid <= `InstValid;
                                end
                                `EXE_MOVN: begin
                                    aluop_o <= `EXE_MOVN_OP;
                                    alusel_o <= `EXE_RES_MOVE;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                    // reg2_o 的值就是地址为rt的通用寄存器的值
                                    if (reg2_o != `ZeroWord) begin
                                        wreg_o <= `WriteEnable;
                                    end else begin
                                        wreg_o <= `WriteDisable;
                                    end
                                end
                                `EXE_MOVZ: begin
                                    aluop_o <= `EXE_MOVZ_OP;
                                    alusel_o <= `EXE_RES_MOVE;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                    // reg2_o 的值就是地址为rt的通用寄存器的值
                                    if (reg2_o == `ZeroWord) begin
                                        wreg_o <= `WriteEnable;
                                    end else begin
                                        wreg_o <= `WriteDisable;
                                    end
                                end
                                `EXE_SLT: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_SLT_OP;
                                    alusel_o <= `EXE_RES_ARITHMETIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_SLTU: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_SLTU_OP;
                                    alusel_o <= `EXE_RES_ARITHMETIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_ADD: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_ADD_OP;
                                    alusel_o <= `EXE_RES_ARITHMETIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_ADDU: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_ADDU_OP;
                                    alusel_o <= `EXE_RES_ARITHMETIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_SUB: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_SUB_OP;
                                    alusel_o <= `EXE_RES_ARITHMETIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_SUBU: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_SUBU_OP;
                                    alusel_o <= `EXE_RES_ARITHMETIC;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_MULT: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_MULT_OP;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_MULTU: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_MULTU_OP;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_DIV: begin
                                    wreg_o <= `WriteDisable;
                                    aluop_o <= `EXE_DIV_OP;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_DIVU: begin
                                    wreg_o <= `WriteDisable;
                                    aluop_o <= `EXE_DIVU_OP;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b1;
                                    instvalid <= `InstValid;
                                end
                                `EXE_JR: begin
                                    wreg_o <= `WriteDisable;
                                    aluop_o <= `EXE_JR_OP;
                                    alusel_o <= `EXE_RES_JUMP_BRANCH;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b0;
                                    link_addr_o <= `ZeroWord;
                                    branch_target_address_o <= reg1_o;
                                    branch_flag_o <= `Branch;
                                    next_inst_in_delayslot_o <= `InDelaySlot;
                                    instvalid <= `InstValid;
                                end
                                `EXE_JALR: begin
                                    wreg_o <= `WriteEnable;
                                    aluop_o <= `EXE_JALR_OP;
                                    alusel_o <= `EXE_RES_JUMP_BRANCH;
                                    reg1_read_o <= 1'b1;
                                    reg2_read_o <= 1'b0;
                                    wd_o <= inst_i[15: 11];
                                    link_addr_o <= pc_plus_8;
                                    branch_target_address_o <= reg1_o;
                                    branch_flag_o <= `Branch;
                                    next_inst_in_delayslot_o <= `InDelaySlot;
                                    instvalid <= `InstValid;
                                end
                                default: begin
                                end
                            endcase
                        end
                        default: begin
                        end
                    endcase  // case op2
                end
                `EXE_LB: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_LB_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_LBU: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_LBU_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_LH: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_LH_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_LHU: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_LHU_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_LWL: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_LWL_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_LWR: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_LWR_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_LL: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_LL_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_SB: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_SB_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    instvalid <= `InstValid;
                end
                `EXE_SC: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_SC_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_SH: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_SH_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    instvalid <= `InstValid;
                end
                `EXE_SW: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_SW_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    instvalid <= `InstValid;
                end
                `EXE_SWL: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_SWL_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    instvalid <= `InstValid;
                end
                `EXE_SWR: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_SWR_OP;
                    alusel_o <= `EXE_RES_LOAD_STORE;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    instvalid <= `InstValid;
                end
                `EXE_ORI: begin
                    // ori 指令需要将结果写入目的寄存器, 所以wreg_o为WriteEnable
                    wreg_o <= `WriteEnable;
                    // 运算的子类型是逻辑"或"运算
                    aluop_o <= `EXE_OR_OP;
                    // 运算类型是逻辑运算
                    alusel_o <= `EXE_RES_LOGIC;
                    // 需要通过Regfile的读端口1读取寄存器
                    reg1_read_o <= 1'b1;
                    // 不需要通过Regfile的读端口2读取寄存器
                    reg2_read_o <= 1'b0;
                    // 指令所需要的立即数
                    imm <= {16'h0, inst_i[15: 0]};
                    // 指令执行要写的目的寄存器地址
                    wd_o <= inst_i[20: 16];
                    // ori指令是有效指令
                    instvalid <= `InstValid;
                end
                `EXE_ANDI: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_AND_OP;
                    alusel_o <= `EXE_RES_LOGIC;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    imm <= {16'h0, inst_i[15: 0]};
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_XORI: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_XOR_OP;
                    alusel_o <= `EXE_RES_LOGIC;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    imm <= {16'h0, inst_i[15: 0]};
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_LUI: begin  // 转化为OR指令
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_OR_OP;
                    alusel_o <= `EXE_RES_LOGIC;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    imm <= {inst_i[15: 0], 16'h0};
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_PREF: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_NOP_OP;
                    alusel_o <= `EXE_RES_NOP;
                    reg1_read_o <= 1'b0;
                    reg2_read_o <= 1'b0;
                    instvalid <= `InstValid;
                end
                `EXE_SLTI: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_SLT_OP;
                    alusel_o <= `EXE_RES_ARITHMETIC;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    imm <= {{16{inst_i[15]}}, inst_i[15: 0]};
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_SLTIU: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_SLTU_OP;
                    alusel_o <= `EXE_RES_ARITHMETIC;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    imm <= {{16{inst_i[15]}}, inst_i[15: 0]};
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_ADDI: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_ADDI_OP;
                    alusel_o <= `EXE_RES_ARITHMETIC;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    imm <= {{16{inst_i[15]}}, inst_i[15: 0]};
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_ADDIU: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_ADDIU_OP;
                    alusel_o <= `EXE_RES_ARITHMETIC;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    imm <= {{16{inst_i[15]}}, inst_i[15: 0]};
                    wd_o <= inst_i[20: 16];
                    instvalid <= `InstValid;
                end
                `EXE_J: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_J_OP;
                    alusel_o <= `EXE_RES_JUMP_BRANCH;
                    reg1_read_o <= 1'b0;
                    reg2_read_o <= 1'b0;
                    link_addr_o <= `ZeroWord;
                    branch_flag_o <= `Branch;
                    next_inst_in_delayslot_o <= `InDelaySlot;
                    instvalid <= `InstValid;
                    branch_target_address_o <= {pc_plus_4[31: 28], inst_i[25: 0], 2'b00};
                end
                `EXE_JAL: begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_JAL_OP;
                    alusel_o <= `EXE_RES_JUMP_BRANCH;
                    reg1_read_o <= 1'b0;
                    reg2_read_o <= 1'b0;
                    wd_o <= 5'b11111;
                    link_addr_o <= pc_plus_8;
                    branch_flag_o <= `Branch;
                    next_inst_in_delayslot_o <= `InDelaySlot;
                    instvalid <= `InstValid;
                    branch_target_address_o <= {pc_plus_4[31: 28], inst_i[25: 0], 2'b00};
                end
                `EXE_BEQ: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_BEQ_OP;
                    alusel_o <= `EXE_RES_JUMP_BRANCH;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    instvalid <= `InstValid;
                    if (reg1_o == reg2_o) begin
                        branch_target_address_o <= pc_plus_4 + imm_sll2_signedext;
                        branch_flag_o <= `Branch;
                        next_inst_in_delayslot_o <= `InDelaySlot;
                    end
                end
                `EXE_BGTZ: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_BGTZ_OP;
                    alusel_o <= `EXE_RES_JUMP_BRANCH;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    instvalid <= `InstValid;
                    if ((reg1_o[31] == 1'b0) && (reg1_o != `ZeroWord)) begin
                        branch_target_address_o <= pc_plus_4 + imm_sll2_signedext;
                        branch_flag_o <= `Branch;
                        next_inst_in_delayslot_o <= `InDelaySlot;
                    end
                end
                `EXE_BLEZ: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_BLEZ_OP;
                    alusel_o <= `EXE_RES_JUMP_BRANCH;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b0;
                    instvalid <= `InstValid;
                    if ((reg1_o[31] == 1'b1) || (reg1_o != `ZeroWord)) begin
                        branch_target_address_o <= pc_plus_4 + imm_sll2_signedext;
                        branch_flag_o <= `Branch;
                        next_inst_in_delayslot_o <= `InDelaySlot;
                    end
                end
                `EXE_BNE: begin
                    wreg_o <= `WriteDisable;
                    aluop_o <= `EXE_BLEZ_OP;
                    alusel_o <= `EXE_RES_JUMP_BRANCH;
                    reg1_read_o <= 1'b1;
                    reg2_read_o <= 1'b1;
                    instvalid <= `InstValid;
                    if (reg1_o !=  reg2_o) begin
                        branch_target_address_o <= pc_plus_4 + imm_sll2_signedext;
                        branch_flag_o <= `Branch;
                        next_inst_in_delayslot_o <= `InDelaySlot;
                    end
                end

                `EXE_REGIMM_INST: begin
                    case (op4)
                        `EXE_BGEZ: begin
                            wreg_o <= `WriteDisable;
                            aluop_o <= `EXE_BGEZ_OP;
                            alusel_o <= `EXE_RES_JUMP_BRANCH;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b0;
                            if (reg1_o[31] == 1'b0) begin
                                branch_target_address_o <= pc_plus_4 + imm_sll2_signedext;
                                branch_flag_o <= `Branch;
                                next_inst_in_delayslot_o <= `InDelaySlot;
                            end
                        end
                        `EXE_BGEZAL: begin
                            wreg_o <= `WriteEnable;
                            aluop_o <= `EXE_BGEZAL_OP;
                            alusel_o <= `EXE_RES_JUMP_BRANCH;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b0;
                            link_addr_o <= pc_plus_8;
                            wd_o <= 5'b11111;
                            instvalid <= `InstValid;
                            if (reg1_o[31] == 1'b0) begin
                                branch_target_address_o <= pc_plus_4 + imm_sll2_signedext;
                                branch_flag_o <= `Branch;
                                next_inst_in_delayslot_o <= `InDelaySlot;
                            end
                        end
                        `EXE_BLTZ: begin
                            wreg_o <= `WriteDisable;
                            aluop_o <= `EXE_BGEZAL_OP;
                            alusel_o <= `EXE_RES_JUMP_BRANCH;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b0;
                            instvalid <= `InstValid;
                            if (reg1_o[31] == 1'b1) begin
                                branch_target_address_o <= pc_plus_4 + imm_sll2_signedext;
                                branch_flag_o <= `Branch;
                                next_inst_in_delayslot_o <= `InDelaySlot;
                            end
                        end
                        `EXE_BLTZAL: begin
                            wreg_o <= `WriteEnable;
                            aluop_o <= `EXE_BGEZAL_OP;
                            alusel_o <= `EXE_RES_JUMP_BRANCH;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b0;
                            link_addr_o <= pc_plus_8;
                            wd_o <= 5'b11111;
                            instvalid <= `InstValid;
                            if (reg1_o[31] == 1'b1) begin
                                branch_target_address_o <= pc_plus_4 + imm_sll2_signedext;
                                branch_flag_o <= `Branch;
                                next_inst_in_delayslot_o <= `InDelaySlot;
                            end
                        end
                        default: begin
                        end
                    endcase
                end

                `EXE_SPECIAL2_INST: begin  // op 等于 SPECIAL2
                    case (op3)
                        `EXE_CLZ: begin
                            wreg_o <= `WriteEnable;
                            aluop_o <= `EXE_CLZ_OP;
                            alusel_o <= `EXE_RES_ARITHMETIC;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b0;
                            instvalid <= `InstValid;
                        end
                        `EXE_CLO: begin
                            wreg_o <= `WriteEnable;
                            aluop_o <= `EXE_CLO_OP;
                            alusel_o <= `EXE_RES_ARITHMETIC;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b0;
                            instvalid <= `InstValid;
                        end
                        `EXE_MUL: begin
                            wreg_o <= `WriteEnable;
                            aluop_o <= `EXE_MUL_OP;
                            alusel_o <= `EXE_RES_MUL;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b1;
                            instvalid <= `InstValid;
                        end
                        `EXE_MADD: begin
                            wreg_o <= `WriteDisable;
                            aluop_o <= `EXE_MADD_OP;
                            alusel_o <= `EXE_RES_MUL;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b1;
                            instvalid <= `InstValid;
                        end
                        `EXE_MADDU: begin
                            wreg_o <= `WriteDisable;
                            aluop_o <= `EXE_MADDU_OP;
                            alusel_o <= `EXE_RES_MUL;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b1;
                            instvalid <= `InstValid;
                        end
                        `EXE_MSUB: begin
                            wreg_o <= `WriteDisable;
                            aluop_o <= `EXE_MSUB_OP;
                            alusel_o <= `EXE_RES_MUL;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b1;
                            instvalid <= `InstValid;
                        end
                        // TODO: EXE_MSUBU指令时序结果不符合预期, 未找到原因
                        `EXE_MSUBU: begin
                            wreg_o <= `WriteDisable;
                            aluop_o <= `EXE_MSUBU_OP;
                            alusel_o <= `EXE_RES_MUL;
                            reg1_read_o <= 1'b1;
                            reg2_read_o <= 1'b1;
                            instvalid <= `InstValid;
                        end
                        default: begin
                        end
                    endcase  // EXE_SPECIAL_INST2 case
                end
                default: begin
                end
            endcase  // case op

            if (inst_i[31: 21] == 11'b000_0000_0000) begin
                if (op3 == `EXE_SLL) begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_SLL_OP;
                    alusel_o <= `EXE_RES_SHIFT;
                    reg1_read_o <= 1'b0;
                    reg2_read_o <= 1'b1;
                    imm[4: 0] <= inst_i[10: 6];
                    wd_o <= inst_i[15: 11];
                    instvalid <= `InstValid;
                end else if (op3 == `EXE_SRL) begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_SRL_OP;
                    alusel_o <= `EXE_RES_SHIFT;
                    reg1_read_o <= 1'b0;
                    reg2_read_o <= 1'b1;
                    imm[4: 0] <= inst_i[10: 6];
                    wd_o <= inst_i[15: 11];
                    instvalid <= `InstValid;
                end else if (op3 == `EXE_SRA) begin
                    wreg_o <= `WriteEnable;
                    aluop_o <= `EXE_SRA_OP;
                    alusel_o <= `EXE_RES_SHIFT;
                    reg1_read_o <= 1'b0;
                    reg2_read_o <= 1'b1;
                    imm[4: 0] <= inst_i[10: 6];
                    wd_o <= inst_i[15: 11];
                    instvalid <= `InstValid;
                end
            end
        end
    end

    // 确定进行运算的源操作数1
    always @(*) begin
        if (rst == `RstEnable) begin
            reg1_o <= `ZeroWord;
        // 如果上一条指令是加载指令, 且该加载指令要加载到的目的寄存器就是当前指令
        // 要通过Regfile模块读端口1读取的通用寄存器,那么表示存在load相关
        // 设置stallreq_for_reg1_loadrelate为Stop
        end else if (pre_inst_is_load == 1'b1 && ex_wd_i == reg1_addr_o && reg1_addr_o == 1'b1) begin
            stallreq_for_reg1_loadrelate <= `Stop;
        end else if((reg1_read_o == 1'b1) && (ex_wreg_i == 1'b1) && (ex_wd_i == reg1_addr_o)) begin
            // regfile模块读端口1要读取的寄存器就是执行阶段要写的目的寄存器, 直接赋值
            reg1_o <= ex_wdata_i;
        end else if((reg1_read_o == 1'b1) && (mem_wreg_i == 1'b1) && (mem_wd_i == reg1_addr_o)) begin
            // regfile模块读端口1要读取的寄存器就是访存阶段要写的目的寄存器, 直接赋值
            reg1_o <= mem_wdata_i;
        end else if (reg1_read_o == 1'b1) begin
            reg1_o <= reg1_data_i;  // Regfile 读端口1的输出值
        end else if (reg1_read_o == 1'b0) begin
            reg1_o <= imm;  // 立即数
        end else begin
            reg1_o <= `ZeroWord;
        end
    end

    // 确定进行运算的源操作数2
    always @(*) begin
        if (rst == `RstEnable) begin
            reg2_o <= `ZeroWord;
        // 如果上一条指令是加载指令, 且该加载指令要加载到的目的寄存器就是当前指令
        // 要通过Regfile模块读端口1读取的通用寄存器,那么表示存在load相关
        // 设置stallreq_for_reg1_loadrelate为Stop
        end else if (pre_inst_is_load == 1'b1 && ex_wd_i == reg2_addr_o && reg2_addr_o == 1'b1) begin
            stallreq_for_reg2_loadrelate <= `Stop;
        end else if((reg2_read_o == 1'b1) && (ex_wreg_i == 1'b1) && (ex_wd_i == reg2_addr_o)) begin
            // regfile模块读端口1要读取的寄存器就是执行阶段要写的目的寄存器, 直接赋值
            reg2_o <= ex_wdata_i;
        end else if((reg2_read_o == 1'b1) && (mem_wreg_i == 1'b1) && (mem_wd_i == reg2_addr_o)) begin
            // regfile模块读端口1要读取的寄存器就是访存阶段要写的目的寄存器, 直接赋值
            reg2_o <= mem_wdata_i;
        end else if (reg2_read_o == 1'b1) begin
            reg2_o <= reg2_data_i;  // Regfile 读端口1的输出值
        end else if (reg2_read_o == 1'b0) begin
            reg2_o <= imm;
        end else begin
            reg2_o <= `ZeroWord;
        end
    end

    // 表示存在load相关, 从而要求流水线暂停, 设置stallreq为stop
    assign stallreq = stallreq_for_reg1_loadrelate | stallreq_for_reg2_loadrelate;

    // 输出变量is_in_delayslot_o表示当前译码阶段指令是否是延迟槽指令
    always @(*) begin
        if (rst == `RstEnable) begin
            is_in_delayslot_o <= `NotInDelaySlot;
        end else begin
            // 直接等于is_in_delayslot_i
            is_in_delayslot_o <= is_in_delayslot_i;
        end
    end


endmodule
