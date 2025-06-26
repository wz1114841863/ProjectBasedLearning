package tutorial

import chisel3._
import circt.stage.ChiselStage

// An FIR Filter
class My4ElementFir(b0: Int, b1: Int, b2: Int, b3: Int) extends Module {
    val io = IO(new Bundle {
        val in = Input(UInt(8.W)) // 输入信号
        val out = Output(UInt(8.W)) // 输出信号
    })
    val x_n1 = RegNext(io.in, 0.U) // 延迟线寄存器
    val x_n2 = RegNext(x_n1, 0.U) // 延迟线寄存器
    val x_n3 = RegNext(x_n2, 0.U) // 延迟线寄存器
    io.out := (b0.U * io.in + b1.U * x_n1 + b2.U * x_n2 + b3.U * x_n3)
}

class MyManyDynamicElementVecFir(length: Int) extends Module {
    val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val valid = Input(Bool())
        val out = Output(UInt(8.W))
        val consts = Input(Vec(length, UInt(8.W)))
    })
    val taps = Seq(io.in) ++ Seq.fill(io.consts.length - 1)(RegInit(0.U(8.W)))
    // TODO:
}

object Main8 extends App {
    println(getVerilogString(new My4ElementFir(1, 2, 3, 4)))
}
