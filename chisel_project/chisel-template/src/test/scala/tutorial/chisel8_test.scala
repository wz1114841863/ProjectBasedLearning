package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers
import scala.util.Random

class My4ElementFirSpec extends AnyFreeSpec with Matchers with ChiselSim {
    "My4Element: {0, 0, 0, 0}" in {
        simulate(new My4ElementFir(0, 0, 0, 0)) { dut =>
            dut.io.in.poke(0.U)
            dut.io.out.expect(0.U)
            dut.clock.step(1)

            dut.io.in.poke(4.U)
            dut.io.out.expect(0.U)
            dut.clock.step(1)

            dut.io.in.poke(5.U)
            dut.io.out.expect(0.U)
            dut.clock.step(1)

            dut.io.in.poke(2.U)
            dut.io.out.expect(0.U)
        }
    }

    "My4Element: {1, 1, 1, 1}" in {
        simulate(new My4ElementFir(1, 1, 1, 1)) { dut =>
            dut.io.in.poke(1.U)
            dut.io.out.expect(1.U) // 1, 0, 0, 0
            dut.clock.step(1)

            dut.io.in.poke(4.U)
            dut.io.out.expect(5.U) // 4, 1, 0, 0
            dut.clock.step(1)

            dut.io.in.poke(3.U)
            dut.io.out.expect(8.U) // 3, 4, 1, 0
            dut.clock.step(1)

            dut.io.in.poke(2.U)
            dut.io.out.expect(10.U) // 2, 3, 4, 1
            dut.clock.step(1)

            dut.io.in.poke(7.U)
            dut.io.out.expect(16.U) // 7, 2, 3, 4
            dut.clock.step(1)

            dut.io.in.poke(0.U)
            dut.io.out.expect(12.U) // 0, 7, 2, 3
        }
    }

    "My4Element: {1, 2, 3, 4}" in {
        simulate(new My4ElementFir(1, 2, 3, 4)) { dut =>
            dut.io.in.poke(1.U)
            dut.io.out.expect(1.U) // 1*1, 0*2, 0*3, 0*4
            dut.clock.step(1)

            dut.io.in.poke(4.U)
            dut.io.out.expect(6.U) // 4*1, 1*2, 0*3, 0*4
            dut.clock.step(1)

            dut.io.in.poke(3.U)
            dut.io.out.expect(14.U) // 3*1, 4*2, 1*3, 0*4
            dut.clock.step(1)

            dut.io.in.poke(2.U)
            dut.io.out.expect(24.U) // 2*1, 3*2, 4*3, 1*4
            dut.clock.step(1)

            dut.io.in.poke(7.U)
            dut.io.out.expect(36.U) // 7*1, 2*2, 3*3, 4*4
            dut.clock.step(1)

            dut.io.in.poke(0.U)
            dut.io.out.expect(32.U) // 0*1, 7*2, 2*3, 3*4
        }
    }
}
