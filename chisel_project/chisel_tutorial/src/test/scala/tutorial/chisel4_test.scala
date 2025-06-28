package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers
import scala.util.Random

class ArbiterSpec extends AnyFreeSpec with Matchers with ChiselSim {
    "Arbiter should select the correct PE" in {
        simulate(new Arbiter()) { dut =>
            val data = Random.nextInt(65536) // 随机生成16位数据
            dut.io.fifo_data.poke(data.U)

            for (i <- 0 until 8) {
                dut.io.fifo_valid.poke((((i >> 0) % 2) != 0).B)
                dut.io.pe0_ready.poke((((i >> 1) % 2) != 0).B)
                dut.io.pe1_ready.poke((((i >> 2) % 2) != 0).B)

                dut.io.fifo_ready.expect((i > 1).B) // FIFO准备信号, 仅在i大于1时为真
                dut.io.pe0_valid.expect((i == 3 || i == 7).B)
                dut.io.pe1_valid.expect((i == 5).B)

                if (i == 3 || i == 7) {
                    dut.io.pe0_data.expect(data.U)
                } else if (i == 5) {
                    dut.io.pe1_data.expect(data.U)
                }
            }
        }
    }
}
