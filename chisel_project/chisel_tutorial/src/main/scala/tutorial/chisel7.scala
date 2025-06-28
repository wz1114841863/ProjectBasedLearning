package tutorial

import chisel3._
import circt.stage.ChiselStage

class RegisterModule extends Module {
    val io = IO(new Bundle {
        val in = Input(UInt(12.W)) // 输入信号
        val out = Output(UInt(12.W)) // 输出信号
    })

    val register = RegInit(UInt(12.W), 0.U) // 定义一个12位寄存器
    register := io.in // 将输入信号赋值给寄存器
    io.out := register // 将寄存器的值输出
}

class MyShiftRegister(val init: Int = 1) extends Module {
    val io = IO(new Bundle {
        val in = Input(Bool())
        val out = Output(UInt(4.W))
    })
    val state = RegInit(UInt(4.W), init.U)
    state := (state << 1) | io.in.asUInt // 左移并将输入信号加入最低位
    io.out := state // 输出当前状态
}

class ClockExamples extends Module {
    val io = IO(new Bundle {
        val in = Input(UInt(10.W))
        val alternateReset = Input(Bool())
        val alternateClock = Input(Clock())
        val outImplicit = Output(UInt(10.W)) // 明确指定10位宽
        val outAlternateReset = Output(UInt(10.W)) // 明确指定10位宽
        val outAlternateClock = Output(UInt(10.W)) // 明确指定10位宽
        val outAlternateBoth = Output(UInt(10.W)) // 明确指定10位宽
    })

    val imp = RegInit(0.U(10.W))
    imp := io.in
    io.outImplicit := imp

    withReset(io.alternateReset) {
        val altRst = RegInit(0.U(10.W))
        altRst := io.in
        io.outAlternateReset := altRst
    }

    withClock(io.alternateClock) {
        val altClk = RegInit(0.U(10.W))
        altClk := io.in
        io.outAlternateClock := altClk
    }

    withClockAndReset(io.alternateClock, io.alternateReset) {
        val alt = RegInit(0.U(10.W))
        alt := io.in
        io.outAlternateBoth := alt
    }
}

object Main7 extends App {
    println(getVerilogString(new RegisterModule()))
    println(getVerilogString(new MyShiftRegister()))
    println(getVerilogString(new ClockExamples()))
}
