package fetch

import chisel3._
import chisel3.util._
import chisel3.util.experimental.loadMemoryFromFile
import common.Consts._
import scala.reflect.internal.Mode
import scala.io.Source

/*
    ImemPortIO:
        addr: 存储器地址输入端口, 可寻址范围是4GB空间
        inst: 指令数据输出端口, 所有标准指令(RV32I)被编码为32位
 */
class ImemPortIO extends Bundle {
    val addr = Input(UInt(WORD_LEN.W))
    val inst = Output(UInt(WORD_LEN.W))
}

class Memory extends Module {
    val io = IO(new Bundle {
        val imem = new ImemPortIO()
    })

    // 生成8bit * 16384(16KB)个寄存器作为存储器实体
    // 选择8bit位宽的原因是PC计数宽度设为4, 一个地址存放8bit
    val mem = Mem(16384, UInt(8.W))

    // 写入 mem
    // 将 hex 文件内容转换为 Chisel 可识别的 Seq[UInt]
    val hexLines = Source.fromFile("src/hex/fetch.hex").getLines().toSeq
    val initData = hexLines.map { line =>
        val trimmedLine = line.trim // 去除首尾空格/制表符
        require(trimmedLine.nonEmpty, s"Empty line in hex file!") // 检查空行
        s"h$trimmedLine".U(8.W) // 添加 "h" 前缀并转换为UInt
    }
    // 初始化 Mem
    initData.zipWithIndex.foreach { case (data, addr) =>
        mem(addr.U) := data
    }
    // 加载存储器数据
    // loadMemoryFromFile(mem, "src/hex/fetch.hex.txt")
    // 打印前n个字节
    // for (i <- 0 until 12) {
    //     printf(p"mem[$i] = ${mem(i.U)}\n")
    // }
    // printf("Finish load hex")

    // 连接4个地址存储的8bit数据, 形成32位数据
    io.imem.inst := Cat(
      mem(io.imem.addr + 3.U(WORD_LEN.W)),
      mem(io.imem.addr + 2.U(WORD_LEN.W)),
      mem(io.imem.addr + 1.U(WORD_LEN.W)),
      mem(io.imem.addr)
    )
}
