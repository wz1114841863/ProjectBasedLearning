package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers
import scala.util.Random

class RegisterSpec extends AnyFreeSpec with Matchers with ChiselSim {
    "Register should hold the input value" in {
        simulate(new RegisterModule()) { dut =>
            for (i <- 0 until 8) {
                val inputValue = Random.nextInt(4096)
                dut.io.in.poke(inputValue.U)
                dut.clock.step(1)
                dut.io.out.expect(inputValue.U)
            }
        }
    }
}

class ShiftRegisterSpec extends AnyFreeSpec with Matchers with ChiselSim {
    "ShiftRegister should shift input and maintain state" in {
        simulate(new MyShiftRegister()) { dut =>
            var state = dut.init
            for (i <- 0 until 10) {
                // poke in LSB of i (i % 2)
                dut.io.in.poke(((i % 2) != 0).B)
                // update expected state
                state = ((state * 2) + (i % 2)) & 0xf
                dut.clock.step(1)
                dut.io.out.expect(state.U)
            }
        }
    }
}


