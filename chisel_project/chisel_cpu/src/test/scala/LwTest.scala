package fetch

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.ConfigMap

class LwTest extends AnyFreeSpec with Matchers with ChiselSim {
    "my cpu should work through hex" in {
        simulate(new Top()) { dut =>
            while (!dut.io.exit.peek().litToBoolean) {
                dut.clock.step(1)
            }
        }
    }
}
