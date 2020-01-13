package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class Scatter[T: ClassTag](batchSize: Int, nOutput: Int)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T] {
  output = Tensor[T]()
  gradInput = T(Tensor[T]())

  protected val inputBuffer: Tensor[Int] = Tensor[Int]()

  override def updateOutput(input: Table): Tensor[T] = {
    val inputTensor = input[Tensor[T]](1)
    val indexTensor = input[Tensor[Int]](2)

    inputBuffer.set(indexTensor.storage(),
      indexTensor.storageOffset(),
      Array(indexTensor.nElement()))

    output.resize(batchSize, nOutput).zero()
    var i = 0
    while (i < inputTensor.nElement()) {
      val index = inputBuffer.valueAt(i + 1)
      require(index < batchSize,
        s"index should smaller than $batchSize, but got $index")
      output.select(1, index + 1).add(inputTensor.select(1, i + 1))
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val inputTensor = input[Tensor[T]](1)
    val indexTensor = input[Tensor[Int]](2)

    val gradTensor = gradInput[Tensor[T]](1)
    gradTensor.resizeAs(inputTensor)

    inputBuffer.set(indexTensor.storage(),
      indexTensor.storageOffset(),
      Array(indexTensor.nElement()))

    var i = 0
    while (i < inputTensor.nElement()) {
      val index = inputBuffer.valueAt(i + 1)
      require(index < batchSize,
        s"index should smaller than $batchSize, but got $index")
      gradTensor.select(1, i + 1).copy(gradOutput.select(1, index + 1))
      i += 1
    }

    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    inputBuffer.set()
    this
  }
}
