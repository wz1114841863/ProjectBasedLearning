package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class PassthroughSpec extends AnyFreeSpec with Matchers with ChiselSim {
    "Passthrough should forward input" in {
        // simulate有两个输入, 一个是模块实例化,
        // 一个是参数为dut的函数, 这里采用了匿名函数和柯里化
        simulate(new Passthrough()) { dut =>
            val testValues = Seq(0, 1, 15)
            testValues.foreach { value =>
                dut.io.in.poke(value.U)
                dut.io.out.expect(value.U)
                println(
                  f"Input = $value%3d, Output = ${dut.io.out.peek().litValue}%3d"
                )
            }
        }
    }
}

class PassthroughGenerator extends AnyFreeSpec with Matchers with ChiselSim {
    "PassthroughGenerator should forward input" in {
        // 测试不同宽度的生成器
        Seq(8, 16).foreach { width =>
            simulate(new PassthoughGenerator(width)) { dut =>
                val testValues = Seq(0, 1, (1 << width) - 1)

                testValues.foreach { value =>
                    dut.io.in.poke(value.U(width.W))
                    dut.io.out.expect(value.U(width.W))

                    println(
                      f"Width = $width, Input = $value%3d, Output = ${dut.io.out.peek().litValue}%3d"
                    )
                }
            }
        }
        println("Success!")
    }
}
