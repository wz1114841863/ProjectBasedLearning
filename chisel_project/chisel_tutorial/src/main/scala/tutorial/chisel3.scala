package tutorial

import chisel3._
import circt.stage.ChiselStage

class ParameterizedAdder(saturate: Boolean) extends Module {
    val io = IO(new Bundle {
        val a = Input(UInt(4.W))
        val b = Input(UInt(4.W))
        val sum = Output(UInt(4.W))
    })
    val sum = io.a +& io.b
    if (saturate) {
        io.sum := Mux(sum > 15.U, 15.U, sum)
    } else {
        io.sum := sum
    }
}
