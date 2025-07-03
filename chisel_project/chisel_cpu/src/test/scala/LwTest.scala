package fetch

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers
import chisel3.simulator.HasSimulator
import chisel3.testing.HasTestingDirectory

import java.nio.file.FileSystems
import scala.reflect.io.Directory

class LwTest extends AnyFreeSpec with Matchers with ChiselSim {

    def verilatorWithVcd = HasSimulator.simulators
        .verilator(verilatorSettings =
            svsim.verilator.Backend.CompilationSettings(
              traceStyle = Some(
                svsim.verilator.Backend.CompilationSettings.TraceStyle
                    .Vcd(traceUnderscore = true, "trace.vcd")
              )
            )
        )

    "my cpu should work through hex" in {
        implicit val vaerilator = verilatorWithVcd

        val vcdFile = FileSystems
            .getDefault()
            .getPath(
              implicitly[HasTestingDirectory].getDirectory.toString,
              "workdir-verilator",
              "trace.vcd"
            )
            .toFile

        vcdFile.delete
        simulate(new Top()) { dut =>
            enableWaves()
            while (!dut.io.exit.peek().litToBoolean) {
                dut.clock.step(1)
            }
        }

        info(s"$vcdFile exists")
        // vcdFile should (exist)
    }
}
