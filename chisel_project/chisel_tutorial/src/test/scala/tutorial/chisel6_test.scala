package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers
import scala.util.Random

class FsmSpec extends AnyFreeSpec with Matchers with ChiselSim {
    "FSM should transition between states based on inputs" in {
        simulate(new Fsm()) { dut =>
            for (state <- 0 to 3) {
                for (coffee <- List(true, false)) {
                    for (idea <- List(true, false)) {
                        for (pressure <- List(true, false)) {
                            dut.io.state.poke(state.U)
                            dut.io.coffee.poke(coffee.B)
                            dut.io.idea.poke(idea.B)
                            dut.io.pressure.poke(pressure.B)

                            val nextState =
                                gradLife(state, coffee, idea, pressure)

                            // 打印调试信息
                            println(
                              f"State: ${dut.io.state.peek().litValue}%d, " +
                                  f"Coffee: ${dut.io.coffee.peek().litToBoolean}%5s, " +
                                  f"Idea: ${dut.io.idea.peek().litToBoolean}%5s, " +
                                  f"Pressure: ${dut.io.pressure.peek().litToBoolean}%5s => " +
                                  f"Next State: ${nextState}%d (HW: ${dut.io.nextState.peek().litValue}%d)"
                            )

                            dut.io.nextState.expect(nextState.U)

                        }
                    }
                }
            }

        }
    }

    def states =
        Map("idle" -> 0.U, "coding" -> 1.U, "writing" -> 2.U, "grad" -> 3.U)

    def gradLife(
        state: Int,
        coffee: Boolean,
        idea: Boolean,
        pressure: Boolean
    ): Int = {
        state match {
            case 0 => if (coffee) 1 else if (idea) 0 else if (pressure) 2 else 0
            case 1 => if (coffee) 1 else if (idea) 2 else if (pressure) 2 else 1
            case 2 => if (coffee) 2 else if (idea) 2 else if (pressure) 3 else 2
            case 3 => 0
        }
    }
}
