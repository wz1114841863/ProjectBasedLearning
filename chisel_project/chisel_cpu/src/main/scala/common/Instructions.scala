package common

import chisel3._
import chisel3.util._

// 使用BitPat对象定义各指令的位列
// 所以成员都为固定值, 因此使用单例对象
object Instructions {
    // 加载/存储
    val LW = BitPat("b?????????????????010?????0000011")
    val SW = BitPat("b?????????????????010?????0100011")

    // 加法
    val ADD = BitPat("b0000000??????????000?????0110011")
    val ADDI = BitPat("b?????????????????000?????0010011")

    // 减法
    val SUB = BitPat("b0100000??????????000?????0110011")

    // 逻辑运算
    val AND = BitPat("b0000000??????????111?????0110011")
    val OR = BitPat("b0000000??????????110?????0110011")
    val XOR = BitPat("b0000000??????????100?????0110011")
    val ANDI = BitPat("b?????????????????111?????0010011")
    val ORI = BitPat("b?????????????????110?????0010011")
    val XORI = BitPat("b?????????????????100?????0010011")

    // 转移
    val SLL = BitPat("b0000000??????????001?????0110011")
    val SRL = BitPat("b0000000??????????101?????0110011")
    val SRA = BitPat("b0100000??????????101?????0110011")
    val SLLI = BitPat("b0000000??????????001?????0010011")
    val SRLI = BitPat("b0000000??????????101?????0010011")
    val SRAI = BitPat("b0100000??????????101?????0010011")

    // 比较
    val SLT = BitPat("b0000000??????????010?????0110011")
    val SLTU = BitPat("b0000000??????????011?????0110011")
    val SLTI = BitPat("b?????????????????010?????0010011")
    val SLTIU = BitPat("b?????????????????011?????0010011")

    // 条件分支
    val BEQ = BitPat("b?????????????????000?????1100011")
    val BNE = BitPat("b?????????????????001?????1100011")
    val BLT = BitPat("b?????????????????100?????1100011")
    val BGE = BitPat("b?????????????????101?????1100011")
    val BLTU = BitPat("b?????????????????110?????1100011")
    val BGEU = BitPat("b?????????????????111?????1100011")

    // 跳转
    val JAL = BitPat("b?????????????????????????1101111")
    val JALR = BitPat("b?????????????????000?????1100111")

    // 立即数加载
    val LUI = BitPat("b?????????????????????????0110111")
    val AUIPC = BitPat("b?????????????????????????0010111")

    // 控制状态寄存器(CSR)操作
    val CSRRW = BitPat("b?????????????????001?????1110011")
    val CSRRWI = BitPat("b?????????????????101?????1110011")
    val CSRRS = BitPat("b?????????????????010?????1110011")
    val CSRRSI = BitPat("b?????????????????110?????1110011")
    val CSRRC = BitPat("b?????????????????011?????1110011")
    val CSRRCI = BitPat("b?????????????????111?????1110011")

    // 发生异常
    val ECALL = BitPat("b00000000000000000000000001110011")

    // 矢量指令
    val VSETVLI = BitPat("b?????????????????111?????1010111")
    val VLE = BitPat("b000000100000?????????????0000111")
    val VSE = BitPat("b000000100000?????????????0100111")
    val VADDVV = BitPat("b0000001??????????000?????1010111")

    // 矢量计数器指令
    val PCNT = BitPat("b000000000000?????110?????0001011")
}
