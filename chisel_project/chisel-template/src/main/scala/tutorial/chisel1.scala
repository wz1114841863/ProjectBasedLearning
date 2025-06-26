package tutorial

import chisel3._
import _root_.circt.stage.ChiselStage  // root用于避免引用冲突


class Passthrough extends Module {
    val io = IO(new Bundle {
        val in = Input(UInt(4.W))
        val out = Output(UInt(4.W))
    })
    io.out := io.in
}

class PassthoughGenerator(width: Int) extends Module {
    val io = IO(new Bundle {
        val in = Input(UInt(width.W))
        val out = Output(UInt(width.W))
    })
    io.out := io.in
}

object Main extends App {
    println(getVerilogString(new Passthrough))
    println(ChiselStage.emitSystemVerilog(
      gen = new Passthrough,
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
    ))
    println(getVerilogString(new PassthoughGenerator(8)))
    println(getVerilogString(new PassthoughGenerator(16)))
}
