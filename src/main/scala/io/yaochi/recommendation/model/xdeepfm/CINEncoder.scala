package io.yaochi.recommendation.model.xdeepfm

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import io.yaochi.recommendation.util.LayerUtil

import scala.collection.mutable.ArrayBuffer


class CINEncoder(batchSize: Int,
                 nFields: Int,
                 embeddingDim: Int,
                 cinDims: Array[Int],
                 mats: Array[Float]) {
  private val shapeModule = buildShapeModule()

  private val cinModules = buildCINModules()

  private val outputModule = buildOutputModule()

  private var cinOutputTable: Table = _

  def forward(input: Tensor[Float]): Tensor[Float] = {
    val x0Tensor = shapeModule.forward(input)
    var xkTensor = x0Tensor
    var inputTable = T.array(Array(x0Tensor, xkTensor))
    val outputTensors = ArrayBuffer[Tensor[Float]]()
    for (cinModule <- cinModules) {
      xkTensor = cinModule.forward(inputTable)
      inputTable = T.array(Array(x0Tensor, xkTensor))
      outputTensors += xkTensor.toTensor[Float]
    }

    cinOutputTable = T.array(outputTensors.toArray)

    outputModule.forward(cinOutputTable)
      .toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val x0Tensor = shapeModule.forward(input)
    var gradTable = outputModule.backward(cinOutputTable, gradOutput).toTable

    null
  }

  private def buildShapeModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields, embeddingDim), Some(false)))
      .add(Transpose(Array((2, 3))))
  }

  private def buildCINModules(): Array[Sequential[Float]] = {
    val modules = ArrayBuffer[Sequential[Float]]()
    var lastCinDim = nFields
    var curOffset = 0
    for (cinDim <- cinDims) {
      modules += buildCINModule(lastCinDim, cinDim, curOffset)
      curOffset += nFields * lastCinDim * cinDim + cinDim
      lastCinDim = cinDim
    }
    modules.toArray
  }

  private def buildCINModule(lastCinDim: Int, cinDim: Int, curOffset: Int): Sequential[Float] = {
    val x0 = Sequential[Float]()
      .add(Reshape(Array(batchSize * embeddingDim, nFields, 1), Some(false)))

    val xk = Sequential[Float]()
      .add(Reshape(Array(batchSize * embeddingDim, lastCinDim, 1), Some(false)))

    Sequential[Float]()
      .add(ParallelTable[Float]().add(x0).add(xk))
      .add(MM(transB = true))
      .add(Reshape(Array(nFields * lastCinDim)))
      .add(LayerUtil.buildLinear(nFields * lastCinDim, cinDim, mats, curOffset))
  }

  private def buildOutputModule(): JoinTable[Float] = {
    JoinTable(1, 3)
  }
}


