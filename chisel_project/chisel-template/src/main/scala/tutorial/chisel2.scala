package tutorial

import chisel3._
import _root_.circt.stage.ChiselStage // root用于避免引用冲突

class MyModule extends Module {
    val io = IO(new Bundle {
        val in = Input(UInt(4.W))
        val out_add = Output(UInt(5.W))
        val out_sub = Output(UInt(4.W))
        val out_mul = Output(UInt(5.W))
    })
    io.out_add := io.in + 1.U
    io.out_sub := io.in - 1.U
    io.out_mul := io.in * 2.U
}

object Main2 extends App {
    println(getVerilogString(new MyModule))
}
