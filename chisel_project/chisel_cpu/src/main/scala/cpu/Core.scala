package fetch

import chisel3._
import chisel3.util._
import common.Consts._
import common.Instructions._

class Core extends Module {
    val io = IO(new Bundle {
        val imem = Flipped(new ImemPortIO()) // 使用Flipped翻转IO
        val dmem = Flipped(new DmemPortIO())
        val exit = Output(Bool()) // 程序处理结束时返回True
    })

    // 生成32bit * 32个寄存器
    // val reg_file = Mem(32, UInt(WORD_LEN.W))
    val reg_file = RegInit(VecInit(Seq.fill(32)(0.U(WORD_LEN.W))))

    // Instruction Fetch: IF stage
    val pc_reg = RegInit(START_ADDR)
    io.imem.addr := pc_reg

    val inst = io.imem.inst
    pc_reg := pc_reg + 4.U // 每个时钟上升沿地址 + 4

    // Instruction Decode: ID stage
    val rs1_addr = inst(19, 15) // 获取寄存器编号
    val rs2_addr = inst(24, 20)
    val wb_addr = inst(11, 7)
    val rs1_data =
        Mux(rs1_addr =/= 0.U(WORD_LEN.W), reg_file(rs1_addr), 0.U(WORD_LEN.W))
    val rs2_data =
        Mux(rs2_addr =/= 0.U(WORD_LEN.W), reg_file(rs2_addr), 0.U(WORD_LEN.W))

    val imm_i = inst(31, 20)
    val imm_i_sext = Cat(Fill(20, imm_i(11)), imm_i) // 符号扩充

    // Execute: EX stage
    val alu_out = MuxCase(
      0.U(WORD_LEN.W),
      Seq(
        (inst === LW) -> (rs1_data + imm_i_sext) // 存储器地址的计算
      )
    )

    // Memory Access Stage
    io.dmem.addr := alu_out

    // WriteBack: WB stage
    val wb_data = io.dmem.rdata
    when(inst === LW) {
        reg_file(wb_addr) := wb_data
    }

    // Debug
    printf(p"pc_reg : 0x${Hexadecimal(pc_reg)}\n")
    printf(p"inst   : 0x${Hexadecimal(inst)}\n")
    printf(p"rs1_addr : $rs1_addr\n")
    printf(p"rs2_addr : $rs2_addr\n")
    printf(p"wb_addr  : $wb_addr\n")
    printf(p"rs1_data : 0x${Hexadecimal(rs1_data)}\n")
    printf(p"rs2_data : 0x${Hexadecimal(rs2_data)}\n")
    printf(p"wb_data   : 0x${Hexadecimal(wb_data)}\n")
    printf(p"dmem.addr : ${io.dmem.addr}\n")
    printf("---------\n")

    // 会导致直接退出, 无法打印最后一条语句的值.
    // io.exit := (inst === 0x34333231.U(WORD_LEN.W))

    val exitReg = RegInit(false.B)
    exitReg := exitReg || (inst === 0x14131211.U(WORD_LEN.W))
    io.exit := exitReg
}

// object MainCore extends App {
//     // 生成Verilog代码
//     println(getVerilogString(new Core))
// }
