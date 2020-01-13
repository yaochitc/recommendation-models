package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class DotProduct2[T: ClassTag](implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T] {
  gradInput = T(Tensor[T](), Tensor[T]())
  @transient
  private var buffer: Tensor[T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    val input1: Tensor[T] = input(1)
    val input2: Tensor[T] = input(2)

    if (buffer == null) {
      buffer = Tensor[T]()
    }
    buffer.resizeAs(input1).cmul(input1, input2)
    output.sum(buffer, 3).squeeze(3)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val input1: Tensor[T] = input(1)
    val input2: Tensor[T] = input(2)
    val size = input1.size()
    size(2) = 1

    if (gradInput.length() != 2) {
      if (!gradInput.contains(1)) {
        gradInput.update(1, Tensor[T]())
      }
      if (!gradInput.contains(2)) {
        gradInput.update(2, Tensor[T]())
      }
    }

    val gw1: Tensor[T] = gradInput(1)
    val gw2: Tensor[T] = gradInput(2)
    gw1.resizeAs(input1).copy(input2)
    gw2.resizeAs(input2).copy(input1)

    val go = gradOutput.view(size).expandAs(input1)
    gw1.cmul(go)
    gw2.cmul(go)

    gradInput
  }

}

object DotProduct2 {
  def apply[@specialized(Float, Double) T: ClassTag]()
                                                    (implicit ev: TensorNumeric[T]): DotProduct2[T] = {
    new DotProduct2[T]()
  }
}