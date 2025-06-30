package fetch

import chisel3._
import chisel3.util._
import common.Consts._

class Core extends Module {
    val io = IO(new Bundle {
        val imem = Flipped(new ImemPortIO()) // 使用Flipped翻转IO
        val exit = Output(Bool()) // 程序处理结束时返回True
    })

    // 生成32bit * 32个寄存器
    val reg_file = Mem(32, UInt(WORD_LEN.W))

    // Instruction Fetch: IF stage
    val pc_reg = RegInit(START_ADDR)
    pc_reg := pc_reg + 4.U // 每个时钟上升沿地址 + 4
    io.imem.addr := pc_reg
    val inst = io.imem.inst

    // Instruction Decode: ID stage
    val rs1_addr = inst(19, 15) // 获取寄存器编号
    val rs2_addr = inst(24, 20)
    val wb_addr = inst(11, 7)
    val rs1_data =
        Mux(rs1_addr =/= 0.U(WORD_LEN.W), reg_file(rs1_addr), 0.U(WORD_LEN.W))
    val rs2_data =
        Mux(rs2_addr =/= 0.U(WORD_LEN.W), reg_file(rs2_addr), 0.U(WORD_LEN.W))

    // Debug
    printf(p"pc_reg : 0x${Hexadecimal(pc_reg)}\n")
    printf(p"inst   : 0x${Hexadecimal(inst)}\n")
    printf(p"rs1_addr : $rs1_addr\n")
    printf(p"rs2_addr : $rs2_addr\n")
    printf(p"wb_addr  : $wb_addr\n")
    printf(p"rs1_data : 0x${Hexadecimal(rs1_data)}\n")
    printf(p"rs2_data : 0x${Hexadecimal(rs2_data)}\n")
    printf("---------\n")

    io.exit := (inst === 0x34333231.U(WORD_LEN.W))
}
