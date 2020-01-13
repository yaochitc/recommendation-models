package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class Gather[T: ClassTag](batchSize: Int, numPairs: Int, embeddingSize: Int)
                         (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {
  output = T()
  gradInput = T(Tensor[T]())

  protected val rowBuffer: Tensor[Int] = Tensor[Int]()
  protected val colBuffer: Tensor[Int] = Tensor[Int]()

  override def updateOutput(input: Table): Table = {
    val inputTensor = input[Tensor[T]](1)
    val rowTensor = input[Tensor[Int]](2)
    val colTensor = input[Tensor[Int]](3)

    rowBuffer.set(rowTensor.storage(),
      rowTensor.storageOffset(),
      Array(rowTensor.nElement()))

    colBuffer.set(colTensor.storage(),
      colTensor.storageOffset(),
      Array(colTensor.nElement()))

    val rowOutput = Tensor[T]().resize(batchSize, numPairs, embeddingSize).zero()
    val colOutput = Tensor[T]().resize(batchSize, numPairs, embeddingSize).zero()

    var i = 0
    while (i < numPairs) {
      val row = rowBuffer.valueAt(i + 1)
      val col = colBuffer.valueAt(i + 1)
      rowOutput.select(2, i + 1).copy(inputTensor.select(2, row + 1))
      colOutput.select(2, i + 1).copy(inputTensor.select(2, col + 1))
      i += 1
    }

    output.update(1, rowOutput)
    output.update(2, colOutput)

    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    val inputTensor = input[Tensor[T]](1)
    val rowTensor = input[Tensor[Int]](2)
    val colTensor = input[Tensor[Int]](3)

    rowBuffer.set(rowTensor.storage(),
      rowTensor.storageOffset(),
      Array(rowTensor.nElement()))

    colBuffer.set(colTensor.storage(),
      colTensor.storageOffset(),
      Array(colTensor.nElement()))

    val gradTensor = gradInput[Tensor[T]](1)
    gradTensor.resizeAs(inputTensor)

    val rowGradTensor = gradOutput[Tensor[T]](1)
    val colGradTensor = gradOutput[Tensor[T]](2)
    var i = 0
    while (i < numPairs) {
      val row = rowBuffer.valueAt(i + 1)
      val col = colBuffer.valueAt(i + 1)
      gradTensor.select(2, row + 1).add(rowGradTensor.select(2, i + 1))
      gradTensor.select(2, col + 1).add(colGradTensor.select(2, i + 1))
      i += 1
    }

    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    rowBuffer.set()
    this
  }
}

object Gather {
  def apply[T: ClassTag, D: ClassTag]
  (batchSize: Int, numPairs: Int, embeddingSize: Int)
  (implicit ev: TensorNumeric[T]): Gather[T] = new Gather(batchSize, numPairs, embeddingSize)
}