package tutorial

import chisel3._
import chisel3.util._
import _root_.circt.stage.ChiselStage

// 使用组合逻辑实现有限状态机
class Fsm extends Module {
    val io = IO(new Bundle {
        val state = Input(UInt(2.W)) // 输入状态
        val coffee = Input(Bool()) // 输入咖啡信号
        val idea = Input(Bool()) // 输入想法信号
        val pressure = Input(Bool()) // 输入压力信号
        val nextState = Output(UInt(2.W)) // 输出下一个状态
    })
    val idle :: coding :: writing :: grad :: Nil = Enum(4) // 定义四个状态

    io.nextState := idle // 默认下一个状态为idle
    when(io.state === idle) {
        when(io.coffee) { io.nextState := coding }
            .elsewhen(io.idea) { io.nextState := idle }
            .elsewhen(io.pressure) { io.nextState := writing }
    }.elsewhen(io.state === coding) {
        when(io.coffee) { io.nextState := coding }
            .elsewhen(io.idea) { io.nextState := writing }
            .elsewhen(io.pressure) { io.nextState := writing }
            .otherwise { io.nextState := coding }
    }.elsewhen(io.state === writing) {
        when(io.coffee) { io.nextState := writing }
            .elsewhen(io.idea) { io.nextState := writing }
            .elsewhen(io.pressure) { io.nextState := grad }
            .otherwise { io.nextState := writing }
    }.elsewhen(io.state === grad) {
        io.nextState := idle
    }
}

object Main6 extends App {
    // 生成Verilog代码
    println(getVerilogString(new Fsm))
}
