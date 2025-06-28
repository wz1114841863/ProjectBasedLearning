package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers
import scala.util.Random

class Max3Spec extends AnyFreeSpec with Matchers with ChiselSim {
    "Max3 should output the maximum of three inputs" in {
        simulate(new Max3()) { dut =>
            for (i <- 0 until 8) {
                val in1 = Random.nextInt(256)
                val in2 = Random.nextInt(256)
                val in3 = Random.nextInt(256)

                dut.io.in1.poke(in1.U)
                dut.io.in2.poke(in2.U)
                dut.io.in3.poke(in3.U)

                val expectedMax = Seq(in1, in2, in3).max
                dut.io.out.expect(expectedMax.U)
            }
        }
    }
}

class Sort4Spec extends AnyFreeSpec with Matchers with ChiselSim {
    "Sort4 should sort four inputs in ascending order" in {
        simulate(new Sort4()) { dut =>
            for (i <- 0 until 8) {
                val in0 = Random.nextInt(256)
                val in1 = Random.nextInt(256)
                val in2 = Random.nextInt(256)
                val in3 = Random.nextInt(256)

                dut.io.in0.poke(in0.U)
                dut.io.in1.poke(in1.U)
                dut.io.in2.poke(in2.U)
                dut.io.in3.poke(in3.U)

                val sorted = Seq(in0, in1, in2, in3).sorted
                dut.io.out0.expect(sorted(0).U)
                dut.io.out1.expect(sorted(1).U)
                dut.io.out2.expect(sorted(2).U)
                dut.io.out3.expect(sorted(3).U)
            }
        }
    }
}
