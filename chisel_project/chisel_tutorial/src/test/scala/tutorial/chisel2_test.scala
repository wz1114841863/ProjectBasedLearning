package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class MyModuleSpec extends AnyFreeSpec with Matchers with ChiselSim {
    "MyModule should perform arithmetic operations" in {
        simulate(new MyModule()) { dut =>
            // 测试加法
            val moduleInput = 3.U(4.W)
            dut.io.in.poke(moduleInput)
            dut.io.out_add.expect(4.U)
            println(
              f"Input = $moduleInput, Add Output = ${dut.io.out_add.peek().litValue}"
            )

            dut.io.out_sub.expect(2.U)
            println(
              f"Input = $moduleInput, Sub Output = ${dut.io.out_sub.peek().litValue}"
            )

            dut.io.out_mul.expect(6.U)
            println(
              f"Input = $moduleInput, Mul Output = ${dut.io.out_mul.peek().litValue}"
            )
        }
    }
}
