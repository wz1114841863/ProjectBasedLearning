package tutorial

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class PassthroughSpec extends AnyFreeSpec with Matchers with ChiselSim {
  "Passthrough should forward input" in {
    // 创建被测模块实例
    simulate(new Passthrough()) { dut =>
      // 测试值序列
      val testValues = Seq(0, 1, 15)

      // 遍历所有测试值
      testValues.foreach { value =>
        // 驱动输入信号
        dut.io.in.poke(value.U)

        // 立即验证输出(组合逻辑)
        dut.io.out.expect(value.U)

        // 可选:打印调试信息
        println(f"Input = $value%3d, Output = ${dut.io.out.peek().litValue}%3d")
      }
    }
  }
}
