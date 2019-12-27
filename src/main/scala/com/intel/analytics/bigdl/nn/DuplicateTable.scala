package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class DuplicateTable[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends DynamicContainer[Tensor[T], Table, T] {
  override def updateOutput(input: Tensor[T]): Table = {
    var i = 0
    while (i < modules.length) {
      output.update(i + 1, modules(i).forward(input))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Table): Tensor[T] = {
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }

    var i = 0
    while (i < modules.length) {
      gradInput.add(modules(i).updateGradInput(input, gradOutput(i + 1)).toTensor)
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Table): Unit = {
    var i = 0
    while (i < modules.length) {
      modules(i).accGradParameters(input, gradOutput(i + 1))
      i += 1
    }
  }

  override def backward(input: Tensor[T], gradOutput: Table): Tensor[T] = {
    val before = System.nanoTime()
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }

    var i = 0
    while (i < modules.length) {
      gradInput.add(modules(i).updateGradInput(input, gradOutput(i + 1)).toTensor)
      i += 1
    }
    backwardTime += System.nanoTime() - before
    gradInput
  }

  override def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    val outputs = ArrayBuffer[ModuleNode[T]]()
    var outputTuple: Array[ModuleNode[T]] = null
    require(startNodes.length == modules.length, s"ParallelTable: " +
      s"startNodes length ${startNodes.length} is more than modules length ${modules.length}")
    for (i <- modules.indices) {
      outputTuple = modules(i).getEndNodes(Array(startNodes(i)))
      outputs ++= outputTuple
    }
    outputs.toArray
  }
}

object DuplicateTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
                                                    (implicit ev: TensorNumeric[T]): DuplicateTable[T] = {
    new DuplicateTable[T]()
  }
}