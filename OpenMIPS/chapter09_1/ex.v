`timescale 1ns / 1ps
`include "defines.v"

/*
    执行阶段: 依据译码阶段的结果, 对源操作数1, 源操作树2进行指定的运算, 执行阶段包括EX, EX/MEM两个模块
*/


module ex(
    input wire rst,

    input wire [`AluOpBus] aluop_i,
    input wire [`AluSelBus] alusel_i,
    input wire [`RegBus] reg1_i,
    input wire [`RegBus] reg2_i,
    input wire [`RegAddrBus] wd_i,
    input wire wreg_i,

    // HILO模块给出的HI/LO寄存器的值
    input wire [`RegBus] hi_i,
    input wire [`RegBus] lo_i,

    // 回写阶段的指令是否要写HI/LO, 用于检测HI/LO寄存器带来的数据相关问题
    input wire [`RegBus] wb_hi_i,
    input wire [`RegBus] wb_lo_i,
    input wire wb_whilo_i,

    // 访存阶段的指令是否要写HI/LO, 用于检测HI/LO寄存器带来的数据相关问题
    input wire [`RegBus] mem_hi_i,
    input wire [`RegBus] mem_lo_i,
    input wire mem_whilo_i,

    input wire [`DoubleRegBus] hilo_temp_i,
    input wire [1: 0] cnt_i,

    // 来自除法模块的输入
    input wire [`DoubleRegBus] div_result_i,
    input wire div_ready_i,

    // 转移指令
    input wire [`RegBus] link_address_i,  // 处于执行阶段的转移指令要保存的返回地址
    input wire is_in_delayslot_i,  // 当前执行阶段的指令是否位于延迟槽

    // 输入接口inst_i, 其值就是当前处于执行阶段的指令
    input wire [`RegBus] inst_i,

    output reg [`RegAddrBus] wd_o,
    output reg wreg_o,
    output reg [`RegBus] wdata_o,

    // 处于执行阶段的指令对于HI/LO寄存器的写操作请求
    output reg [`RegBus] hi_o,
    output reg [`RegBus] lo_o,
    output reg whilo_o,

    // 发送流水线请求
    output reg stallreq,

    output reg [`DoubleRegBus] hilo_temp_o,
    output reg [1: 0] cnt_o,

    // 到除法模块的输出
    output reg [`RegBus] div_opdata1_o,
    output reg [`RegBus] div_opdata2_o,
    output reg div_start_o,
    output reg signed_div_o,

    // 传递给加载/存储指令
    output wire [`AluOpBus] aluop_o,
    output wire [`RegBus] mem_addr_o,
    output wire [`RegBus] reg2_o
);
    reg [`RegBus] logicout;  // 保存逻辑运算的结果
    reg [`RegBus] shiftres;  // 保存移位运算结果
    reg [`RegBus] moveres;  // 移动操作的结果
    reg [`RegBus] HI;  // 保存HI寄存器的最新值
    reg [`RegBus] LO;  // 保存LO寄存器的最新值

    wire ov_sum;  // 保存溢出情况
    wire reg1_eq_reg2;  // 第一个操作数是否等于第二个操作数
    wire reg1_lt_reg2;  // 第一个操作数是否小于第二个操作数

    reg [`RegBus] arithmeticres;  // 保存算术运算的结果
    wire [`RegBus] reg2_i_mux;  // 保存输入的第二个操作数reg2_i的补码
    wire [`RegBus] reg1_i_not;  // 保存输入的第一个reg1_i取反后的值
    wire [`RegBus] result_sum;  // 保存加法结果
    wire [`RegBus] opdata1_mult;  // 乘法操作中的被乘数
    wire [`RegBus] opdata2_mult;  // 乘法操作中的乘数
    wire [`DoubleRegBus] hilo_temp;  // 临时保存乘法结果, 宽度为64位
    reg [`DoubleRegBus] mulres;  // 保存乘法结果,宽度为64位

    reg [`DoubleRegBus] hilo_temp1;
    reg stallreq_for_madd_msub;

    // 是否由于除法运算导致流水线暂停
    reg stallreq_for_div;

    // aluop_o 会传递到访存阶段, 届时将利用其确定加载, 存储类型
    assign aluop_o = aluop_i;

    // mem_addr_o 会传递到访存阶段, 是加载, 存储指令对应的寄存器地址
    // 此处的reg1_i就是加载和存储指令中地址为base的通用寄存器的值
    // inst_i[15: 0]就是指令中的offset
    assign mem_addr_o = reg1_i + {{16{inst_i[15]}}, inst_i[15: 0]};

    // reg2_i 是存储指令要存储的数据, 或者lwl,lwr指令要加载到目的寄存器的原始值
    assign reg2_o = reg2_i;

    // 计算以下五个变量的值
    // 1. 如果是减法或者有符号比较运算, 那么reg2_i_mux等于第二个操作数reg2_i的补码, 否则等于它本身
    assign reg2_i_mux = ((aluop_i == `EXE_SUB_OP)  ||
                         (aluop_i == `EXE_SUBU_OP) ||
                         (aluop_i == `EXE_SLT_OP)) ?
                         (~reg2_i) + 1: reg2_i;

    // 2. 分情况讨论, reg2_i_mux取决于操作类型, result_sum为加减法的结果
    assign result_sum = reg1_i + reg2_i_mux;

    // 3. 指示是否溢出
    // 1) reg1_i 是正数, reg2_i_mux是正数, 但是两者之和是负数
    // 2) reg1_i 是负数, reg2_i_mux是负数, 但是两者之和是正数
    assign ov_sum = ((!reg1_i[31] && !reg2_i_mux[31]) && result_sum[31]) ||
                     ((reg1_i[31] && reg2_i_mux[31]) && (!result_sum[31]));

    // 4. 计算操作数1是否小于操作数2, 分两种情况
    // 1) 有符号数比较
    // 2) 无符号数比较
    assign reg1_lt_reg2 = (aluop_i == `EXE_SLT_OP) ?
                          ((reg1_i[31] && !reg2_i[31]) ||  // reg1_i是负数, reg2_i是正数
                          (!reg1_i[31] && !reg2_i[31] && result_sum[31]) ||  // 两个都是正数,但是result_sum是负数
                          (reg1_i[31] && reg2_i[31] && result_sum[31])) :
                          (reg1_i < reg2_i);  // 无符号数直接使用比较运算符

    // 对操作数1逐位取反, 赋给reg1_i_not
    assign reg1_i_not = ~reg1_i;

    // 给arithmeticres变量赋值
    always @(*) begin
        if (rst == `RstEnable) begin
            arithmeticres <= `ZeroWord;
        end else begin
            case (aluop_i)
                `EXE_SLT_OP, `EXE_SLTU_OP: begin
                    arithmeticres <= reg1_lt_reg2;  // 比较运算
                end
                `EXE_ADD_OP, `EXE_ADDU_OP, `EXE_ADDI_OP, `EXE_ADDIU_OP: begin
                    arithmeticres <= result_sum;  // 加法运算
                end
                `EXE_SUB_OP, `EXE_SUBU_OP: begin
                    arithmeticres <= result_sum;  // 减法运算
                end
                `EXE_CLZ_OP: begin  // 计数运算clz
                    arithmeticres <=  reg1_i[31] ? 0 : reg1_i[30] ? 1 : reg1_i[29] ? 2 :
                                    reg1_i[28] ? 3 : reg1_i[27] ? 4 : reg1_i[26] ? 5 :
                                    reg1_i[25] ? 6 : reg1_i[24] ? 7 : reg1_i[23] ? 8 :
                                    reg1_i[22] ? 9 : reg1_i[21] ? 10 : reg1_i[20] ? 11 :
                                    reg1_i[19] ? 12 : reg1_i[18] ? 13 : reg1_i[17] ? 14 :
                                    reg1_i[16] ? 15 : reg1_i[15] ? 16 : reg1_i[14] ? 17 :
                                    reg1_i[13] ? 18 : reg1_i[12] ? 19 : reg1_i[11] ? 20 :
                                    reg1_i[10] ? 21 : reg1_i[9] ? 22 : reg1_i[8] ? 23 :
                                    reg1_i[7] ? 24 : reg1_i[6] ? 25 : reg1_i[5] ? 26 :
                                    reg1_i[4] ? 27 : reg1_i[3] ? 28 : reg1_i[2] ? 29 :
                                    reg1_i[1] ? 30 : reg1_i[0] ? 31 : 32;
                end
                `EXE_CLO_OP: begin
                    arithmeticres <= (reg1_i_not[31] ? 0 : reg1_i_not[30] ? 1 : reg1_i_not[29] ? 2 :
                                    reg1_i_not[28] ? 3 : reg1_i_not[27] ? 4 : reg1_i_not[26] ? 5 :
                                    reg1_i_not[25] ? 6 : reg1_i_not[24] ? 7 : reg1_i_not[23] ? 8 :
                                    reg1_i_not[22] ? 9 : reg1_i_not[21] ? 10 : reg1_i_not[20] ? 11 :
                                    reg1_i_not[19] ? 12 : reg1_i_not[18] ? 13 : reg1_i_not[17] ? 14 :
                                    reg1_i_not[16] ? 15 : reg1_i_not[15] ? 16 : reg1_i_not[14] ? 17 :
                                    reg1_i_not[13] ? 18 : reg1_i_not[12] ? 19 : reg1_i_not[11] ? 20 :
                                    reg1_i_not[10] ? 21 : reg1_i_not[9] ? 22 : reg1_i_not[8] ? 23 :
                                    reg1_i_not[7] ? 24 : reg1_i_not[6] ? 25 : reg1_i_not[5] ? 26 :
                                    reg1_i_not[4] ? 27 : reg1_i_not[3] ? 28 : reg1_i_not[2] ? 29 :
                                    reg1_i_not[1] ? 30 : reg1_i_not[0] ? 31 : 32) ;
                end
                default: begin
                    arithmeticres <= `ZeroWord;
                end
            endcase
        end
    end

    // 输出DIV模块控制信息, 获取DIV模块给出的结果
    always @(*) begin
        if (rst == `RstEnable) begin
            stallreq_for_div <= `NoStop;
            div_opdata1_o <= `ZeroWord;
            div_opdata2_o <= `ZeroWord;
            div_start_o <= `DivStop;
            signed_div_o <= 1'b0;
        end else begin
            stallreq_for_div <= `NoStop;
            div_opdata1_o <= `ZeroWord;
            div_opdata2_o <= `ZeroWord;
            div_start_o <= `DivStop;
            signed_div_o <= 1'b0;
            case (aluop_i)
                `EXE_DIV_OP: begin
                    if (div_ready_i == `DivResultNotReady) begin
                        stallreq_for_div <= `Stop;  // 请求流水线暂停信号
                        div_opdata1_o <= reg1_i;  // 被除数
                        div_opdata2_o <= reg2_i;  // 除数
                        div_start_o <= `DivStart;  // 开始除法信号
                        signed_div_o <= 1'b1;  // 有符号除法
                    end else if (div_ready_i == `DivResultReady) begin
                        stallreq_for_div <= `NoStop;  // 请求流水线暂停信号
                        div_opdata1_o <= reg1_i;  // 被除数
                        div_opdata2_o <= reg2_i;  // 除数
                        div_start_o <= `DivStop;  // 开始除法信号
                        signed_div_o <= 1'b1;  // 有符号除法
                    end else begin
                        stallreq_for_div <= `NoStop;
                        div_opdata1_o <= `ZeroWord;
                        div_opdata2_o <= `ZeroWord;
                        div_start_o <= `DivStop;
                        signed_div_o <= 1'b0;
                    end
                end
                `EXE_DIVU_OP: begin
                    if (div_ready_i == `DivResultNotReady) begin
                        stallreq_for_div <= `Stop;  // 请求流水线暂停信号
                        div_opdata1_o <= reg1_i;  // 被除数
                        div_opdata2_o <= reg2_i;  // 除数
                        div_start_o <= `DivStart;  // 开始除法信号
                        signed_div_o <= 1'b0;  // 无符号除法
                    end else if (div_ready_i == `DivResultReady) begin
                        stallreq_for_div <= `NoStop;  // 请求流水线暂停信号
                        div_opdata1_o <= reg1_i;  // 被除数
                        div_opdata2_o <= reg2_i;  // 除数
                        div_start_o <= `DivStop;  // 开始除法信号
                        signed_div_o <= 1'b0;  // 有符号除法
                    end else begin
                        stallreq_for_div <= `NoStop;
                        div_opdata1_o <= `ZeroWord;
                        div_opdata2_o <= `ZeroWord;
                        div_start_o <= `DivStop;
                        signed_div_o <= 1'b0;
                    end
                end
                default: begin
                end
            endcase
        end
    end

    // 1. 取得乘法运算的被乘数, 如果是有符号乘法且被乘数是负数, 那么取补码
    assign opdata1_mult = (((aluop_i == `EXE_MUL_OP) ||
                            (aluop_i == `EXE_MULT_OP) ||
                            (aluop_i == `EXE_MADD_OP) ||
                            (aluop_i == `EXE_MSUB_OP)) &&
                            (reg1_i[31] == 1'b1)) ?
                            (~reg1_i + 1) : reg1_i;

    // 2. 取得乘法运算的乘数, 如果是有符号乘法且乘数是负数,那么取补码
    assign opdata2_mult = (((aluop_i == `EXE_MUL_OP) ||
                            (aluop_i == `EXE_MULT_OP) ||
                            (aluop_i == `EXE_MADD_OP) ||
                            (aluop_i == `EXE_MSUB_OP)) &&
                            (reg2_i[31] == 1'b1)) ?
                            (~reg2_i + 1) : reg2_i;

    // 3. 得到临时乘法结果, 保存在变量hilo_temp中
    assign hilo_temp = opdata1_mult * opdata2_mult;

    // 4. 对临时乘法结果进行修正, 最终的乘法结果保存在变量mulres中,主要是有两点:
    // 1) 如果是有符号乘法指令mult和mul, 那么需要修正临时乘法结果, 如下:
    //    i) 如果异号乘法, 需要对临时乘法结果求补码
    //    ii) 如果是同号乘法, 直接赋值
    // 2) 如果是无符号乘法指令, 直接赋值
    always @(*) begin
        if (rst == `RstEnable) begin
            mulres <= {`ZeroWord, `ZeroWord};
        end else if ((aluop_i == `EXE_MULT_OP) || (aluop_i == `EXE_MUL_OP) ||
                     (aluop_i == `EXE_MADD_OP) || (aluop_i == `EXE_MSUB_OP)) begin
            if (reg1_i[31] ^ reg2_i[31] == 1'b1) begin  // 两个数异号
                mulres <= ~hilo_temp + 1;
            end else begin
                mulres <= hilo_temp;
            end
        end else begin
            mulres <= hilo_temp;
        end
    end

    // MADD, MADDU, MSUB, MSUBU指令
    always @(*) begin
        if (rst == `RstEnable) begin
            hilo_temp_o <= {`ZeroWord, `ZeroWord};
            cnt_o <= 2'b00;
            stallreq_for_madd_msub <= `NoStop;
        end else begin
            case (aluop_i)
                `EXE_MADD_OP, `EXE_MADDU_OP: begin  // madd, maddu指令
                    if (cnt_i == 2'b00) begin  // 执行阶段第一个时钟周期
                        hilo_temp_o <= mulres;
                        cnt_o <= 2'b01;
                        hilo_temp1 <= {`ZeroWord, `ZeroWord};
                        stallreq_for_madd_msub <= `Stop;
                    end else if (cnt_i == 2'b01) begin  // 执行阶段第二个时钟周期
                        hilo_temp_o <= {`ZeroWord, `ZeroWord};
                        cnt_o <= 2'b10;
                        hilo_temp1 <= hilo_temp_i + {HI, LO};
                        stallreq_for_madd_msub <= `NoStop;
                    end
                end
                `EXE_MSUB_OP, `EXE_MSUBU_OP: begin  // msub, msubu指令
                    if (cnt_i == 2'b00) begin  // 执行阶段第一个时钟周期
                        hilo_temp_o <= ~mulres + 1;
                        cnt_o <= 2'b01;
                        stallreq_for_madd_msub <= `Stop;
                        hilo_temp1 <= {`ZeroWord,`ZeroWord};
                    end else if (cnt_i == 2'b01) begin  // 执行阶段第二个时钟周期
                        hilo_temp_o <= {`ZeroWord, `ZeroWord};
                        cnt_o <= 2'b10;
                        hilo_temp1 <= hilo_temp_i + {HI, LO};
                        stallreq_for_madd_msub <= `NoStop;
                    end
                end
                default: begin
                    hilo_temp_o <= {`ZeroWord, `ZeroWord};
                    cnt_o <= 2'b00;
                    stallreq_for_madd_msub <= `NoStop;
                end
            endcase
        end
    end

    // 暂停流水线
    // 乘累加, 乘累减指令会导致流水线暂停
    // DIV会导致流水线暂停
    always @(*) begin
        stallreq = stallreq_for_madd_msub || stallreq_for_div;
    end

    // 得到最新的HI/LO寄存器的值, 此处要解决数据相关问题
    always @(*) begin
        if (rst == `RstEnable) begin
            {HI, LO} <= {`ZeroWord, `ZeroWord};
        end else if (mem_whilo_i == `WriteEnable) begin
            // 访存阶段的指令要写HI/LO寄存器
            {HI, LO} <= {mem_hi_i, mem_lo_i};
        end else if (wb_whilo_i == `WriteEnable) begin
            // 回写阶段的指令要写HI/LO寄存器
            {HI, LO} <= {wb_hi_i, wb_lo_i};
        end else begin
            {HI, LO} <= {hi_i, lo_i};
        end
    end

    // 依据aluop_i指示的运算子类型进行运算
    // 逻辑运算
    always @(*) begin
        if (rst == `RstEnable) begin
            logicout <= `ZeroWord;
        end else begin
            case (aluop_i)
                `EXE_OR_OP: begin
                    logicout <= reg1_i | reg2_i;
                end
                `EXE_AND_OP: begin
                    logicout <= reg1_i & reg2_i;
                end
                `EXE_NOR_OP: begin
                    logicout <= ~(reg1_i | reg2_i);
                end
                `EXE_XOR_OP: begin
                    logicout <= reg1_i ^ reg2_i;
                end
                default: begin
                    logicout <= `ZeroWord;
                end
            endcase
        end
    end

    // 依据aluop_i指示的运算子类型进行运算
    // 移位运算
    always @(*) begin
        if (rst == `RstEnable) begin
            shiftres <= `ZeroWord;
        end else begin
            case (aluop_i)
                `EXE_SLL_OP: begin
                    shiftres <= reg2_i << reg1_i[4: 0];
                end
                `EXE_SRL_OP: begin
                    shiftres <= reg2_i >> reg1_i[4: 0];
                end
                `EXE_SRA_OP: begin
                    shiftres <= ({32{reg2_i[31]}} << (6'd32 - {1'b0, reg1_i[4: 0]})) | reg2_i >> reg1_i[4: 0];
                end
                default: begin
                    shiftres <= `ZeroWord;
                end
            endcase
        end
    end

    // 依据aluop_i指示的运算子类型进行运算
    // 移动运算
    always @(*) begin
        if (rst == `RstEnable) begin
            moveres <= `ZeroWord;
        end else begin
            moveres <= `ZeroWord;
            case (aluop_i)
                `EXE_MFHI_OP: begin
                    // 如果是mfhi指令, 那么将HI的值作为移动操作的结果
                    moveres <= HI;
                end
                `EXE_MFLO_OP: begin
                    // 如果是mflo指令,那么将LO的值作为移动操作的结果
                    moveres <= LO;
                end
                `EXE_MOVZ_OP: begin
                    // 如果movz指令, 那么将reg1_i的值作为移动操作的结果
                    moveres <= reg1_i;
                end
                `EXE_MOVN_OP: begin
                    // 如果movn指令, 那么将reg1_i的值作为移动操作的结果
                    moveres <= reg1_i;
                end
                default: begin
                end
            endcase
        end
    end

    // 依据alusel_i指示的运算类型,选择一个运算结果作为最终结果,此处只有逻辑运算结果
    always @(*) begin
        wd_o <= wd_i;
        // 如果特定指令发生溢出, 那么设置wreg_o为WriteDisable, 表示不写目的寄存器
        if (((aluop_i == `EXE_ADD_OP) || (aluop_i == `EXE_ADDI_OP) ||
             (aluop_i == `EXE_SUB_OP)) && (ov_sum == 1'b1)) begin
            wreg_o <= `WriteDisable;
        end else begin
            wreg_o <= wreg_i;
        end

        case (alusel_i)
            `EXE_RES_LOGIC: begin
                wdata_o  <= logicout;
            end
            `EXE_RES_SHIFT: begin
                wdata_o <= shiftres;
            end
            `EXE_RES_MOVE: begin
                wdata_o <= moveres;
            end
            `EXE_RES_ARITHMETIC: begin  // 除乘法外的简单算术操作指令
                wdata_o <= arithmeticres;
            end
            `EXE_RES_MUL: begin  // 乘法指令mul
                wdata_o <= mulres[31: 0];
            end
            `EXE_RES_JUMP_BRANCH: begin
                wdata_o <= link_address_i;
            end
            default: begin
                wdata_o <= `ZeroWord;
            end
        endcase
    end

    // 修改HI, LO寄存器的写信息, 需要给出whilo_o, hi_o, lo_i的值
    always @(*) begin
        if (rst == `RstEnable) begin
            whilo_o <= `WriteDisable;
            hi_o <= `ZeroWord;
            lo_o <= `ZeroWord;
        end else if ((aluop_i == `EXE_MSUB_OP) || (aluop_i == `EXE_MSUBU_OP)) begin
            whilo_o <= `WriteEnable;
            hi_o <= hilo_temp1[63: 32];
            lo_o <= hilo_temp1[31: 0];
        end else if ((aluop_i == `EXE_MADD_OP) || (aluop_i == `EXE_MADDU_OP)) begin
            whilo_o <= `WriteEnable;
            hi_o <= hilo_temp1[63: 32];
            lo_o <= hilo_temp1[31: 0];
        end else if ((aluop_i == `EXE_MULT_OP) || (aluop_i == `EXE_MULTU_OP)) begin
            whilo_o <= `WriteEnable;
            hi_o <= mulres[63: 32];
            lo_o <= mulres[31: 0];
        end else if ((aluop_i == `EXE_DIV_OP) || (aluop_i == `EXE_DIVU_OP)) begin
            whilo_o <= `WriteEnable;
            hi_o <= div_result_i[63: 32];
            lo_o <= div_result_i[31: 0];
        end else if (aluop_i == `EXE_MTHI_OP) begin
            whilo_o <= `WriteEnable;
            hi_o <= reg1_i;
            lo_o <= LO;  // 写HI寄存器, 所以LO值不变,直接赋值
        end else if (aluop_i == `EXE_MTLO_OP) begin
            whilo_o <= `WriteEnable;
            hi_o <= HI;
            lo_o <= reg1_i;
        end else begin
            whilo_o <= `WriteDisable;
            hi_o <= `ZeroWord;
            lo_o <= `ZeroWord;
        end
    end
endmodule
