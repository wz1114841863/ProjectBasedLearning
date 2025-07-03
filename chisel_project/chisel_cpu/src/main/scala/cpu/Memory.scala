package fetch

import chisel3._
import chisel3.util._
import chisel3.util.experimental.loadMemoryFromFile

import scala.reflect.internal.Mode
import scala.io.Source

import common.Consts._
import common.Instructions._

/*
    ImemPortIO:
        addr: 存储器地址输入端口, 可寻址范围是4GB空间
        inst: 指令数据输出端口, 所有标准指令(RV32I)被编码为32位
 */
class ImemPortIO extends Bundle {
    val addr = Input(UInt(WORD_LEN.W))
    val inst = Output(UInt(WORD_LEN.W))
}

class DmemPortIO extends Bundle {
    val addr = Input(UInt(WORD_LEN.W))
    val rdata = Output(UInt(WORD_LEN.W))
}

class Memory extends Module {
    val io = IO(new Bundle {
        val imem = new ImemPortIO()
        val dmem = new DmemPortIO()
    })

    // 生成8bit * 16384(16KB)个寄存器作为存储器实体
    // 选择8bit位宽的原因是PC计数宽度设为4, 一个地址存放8bit
    val mem = Mem(16384, UInt(8.W))

    // 加载存储器数据, 存在问题, 无法加载
    // loadMemoryFromFile(mem, "src/hex/fetch.hex.txt")
    // 写入 mem
    // 将 hex 文件内容转换为 Chisel 可识别的 Seq[UInt]
    // 在测试或模块代码中
    // val projectRoot = sys.props.getOrElse(
    //   "chisel.project.root",
    //   sys.env.getOrElse("PWD", ".")
    // )
    // val hexFilePath = s"$projectRoot/src/hex/fetch.hex"
    val hexFilePath = s"src/hex/fetch.hex"
    val hexLines = Source.fromFile(hexFilePath).getLines().toSeq
    val initData = hexLines.map { line =>
        val trimmedLine = line.trim
        require(trimmedLine.nonEmpty, s"Empty line in hex file!")
        s"h$trimmedLine".U(8.W)
    }
    // 初始化 Mem
    initData.zipWithIndex.foreach { case (data, addr) =>
        mem(addr.U) := data
    }

    // 连接4个地址存储的8bit数据, 形成32位指令
    io.imem.inst := Cat(
      mem(io.imem.addr + 3.U(WORD_LEN.W)),
      mem(io.imem.addr + 2.U(WORD_LEN.W)),
      mem(io.imem.addr + 1.U(WORD_LEN.W)),
      mem(io.imem.addr)
    )

    // 32位数据
    io.dmem.rdata := Cat(
      mem(io.dmem.addr + 3.U(WORD_LEN.W)),
      mem(io.dmem.addr + 2.U(WORD_LEN.W)),
      mem(io.dmem.addr + 1.U(WORD_LEN.W)),
      mem(io.dmem.addr)
    )
}

// object MainMemory extends App {
//     // 生成Verilog代码
//     println(getVerilogString(new Memory))
// }
