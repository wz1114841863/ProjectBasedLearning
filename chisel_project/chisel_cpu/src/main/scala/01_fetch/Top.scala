package fetch

import chisel3._
import chisel3.util._

class Top extends Module {
    val io = IO(new Bundle {
        val exit = Output(Bool())
    })

    // 用new将Core和Memory类实例化, 并用Module硬件化
    val core = Module(new Core())
    val memory = Module(new Memory())

    // 使用<>批量连接端口
    core.io.imem <> memory.io.imem
    core.io.dmem <> memory.io.dmem

    io.exit := core.io.exit
}

object MainTop extends App {
    // 生成Verilog代码
    println(getVerilogString(new Top))
}
