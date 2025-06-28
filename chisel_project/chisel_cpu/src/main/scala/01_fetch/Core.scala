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

    // 取值
    val pc_reg = RegInit(START_ADDR)
    pc_reg := pc_reg + 4.U // 每个时钟上升沿地址 + 4
    io.imem.addr := pc_reg
    val inst = io.imem.inst

    io.exit := (inst === 0x34333231.U(WORD_LEN.W))

    // Debug
    printf(p"pc_reg : 0x${Hexadecimal(pc_reg)}\n")
    printf(p"inst   : 0x${Hexadecimal(inst)}\n")
    printf("---------\n")
}
