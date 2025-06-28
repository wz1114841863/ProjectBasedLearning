package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers
import scala.util.Random
import scala.math.min

class ParameterizedAdderSpec extends AnyFreeSpec with Matchers with ChiselSim {
    "ParameterizedAdder should correctly add two numbers" in {
        for (saturate <- Seq(true, false)) {
            simulate(new ParameterizedAdder(saturate)) { dut =>
                val cycles = 100
                for (i <- 0 until cycles) {
                    val in_a = Random.nextInt(16)
                    val in_b = Random.nextInt(16)
                    dut.io.a.poke(in_a.U)
                    dut.io.b.poke(in_b.U)
                    if (saturate) {
                        dut.io.sum.expect(min(in_a + in_b, 15).U)
                    } else {
                        dut.io.sum.expect(((in_a + in_b) % 16).U)
                    }
                }

                dut.io.a.poke(15.U)
                dut.io.b.poke(15.U)
                if (saturate) {
                    dut.io.sum.expect(15.U)
                } else {
                    dut.io.sum.expect(14.U)
                }
            }
        }
    }
}
