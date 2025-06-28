package fetch

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class FetchTest extends AnyFreeSpec with Matchers with ChiselSim {
    "my cpu should work through hex" in {
        simulate(new Top()) { dut =>
            var counter = 0
            while (!dut.io.exit.peek().litToBoolean && counter < 10) {
                dut.clock.step(1)
                counter += 1
            }
        }
    }
}
