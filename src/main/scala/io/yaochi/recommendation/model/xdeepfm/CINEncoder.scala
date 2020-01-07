package io.yaochi.recommendation.model.xdeepfm

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import io.yaochi.recommendation.util.LayerUtil

import scala.collection.mutable.ArrayBuffer


class CINEncoder(batchSize: Int,
                 nFields: Int,
                 embeddingDim: Int,
                 fcDims: Array[Int],
                 cinDims: Array[Int],
                 mats: Array[Float],
                 start: Int = 0) {
  private val shapeModule = buildShapeModule()

  private val dnnLinearLayers = buildLinearLayers()

  private val dnnModule = buildDNNModules()

  private val cinModules = buildCINModules()

  private val outputLinearLayer = buildOutputLinearLayer()

  private val outputModule = buildOutputModule()

  private var outputTable: Table = _

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

    outputTensors += dnnModule.forward(input).toTensor[Float]

    outputTable = T.array(outputTensors.toArray)

    outputModule.forward(outputTable)
      .toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutputTable: Table): Tensor[Float] = {
    null
  }

  private def buildShapeModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields, embeddingDim), Some(false)))
      .add(Transpose(Array((2, 3))))
  }

  private def buildDNNModules(): Sequential[Float] = {
    val encoder = Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields * embeddingDim), Some(false)))

    for (linearLayer <- dnnLinearLayers) {
      encoder.add(linearLayer)
      encoder.add(ReLU())
    }
    encoder
  }

  private def buildLinearLayers(): Array[Linear[Float]] = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = start
    var dim = nFields * embeddingDim
    for (fcDim <- fcDims) {
      layers += LayerUtil.buildLinear(dim, fcDim, mats, true, curOffset)
      curOffset += dim * fcDim + fcDim
      dim = fcDim
    }
    layers.toArray
  }

  private def buildCINModules(): Array[Sequential[Float]] = {
    val modules = ArrayBuffer[Sequential[Float]]()
    var lastCinDim = nFields
    var curOffset = start + getDNNParameterSize
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
      .add(LayerUtil.buildLinear(nFields * lastCinDim, cinDim, mats, true, curOffset))
  }

  private def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(JoinTable(1, 3))
      .add(outputLinearLayer)
  }

  private def buildOutputLinearLayer(): Linear[Float] = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = start + getDNNParameterSize + getCINParameterSize
    var dim = cinDims.sum + fcDims.last
    LayerUtil.buildLinear(dim, 1, mats, true, curOffset)
  }

  private def getDNNParameterSize: Int = {
    val dims = Array(nFields * embeddingDim) ++ fcDims
    (1 until dims.length)
      .map(i => dims(i - 1) * dims(i) + dims(i))
      .sum
  }

  private def getCINParameterSize: Int = {
    val dims = Array(nFields) ++ cinDims
    (1 until dims.length)
      .map(i => nFields * dims(i - 1) * dims(i) + dims(i))
      .sum
  }
}

object CINEncoder {
  def apply(batchSize: Int,
            nFields: Int,
            embeddingDim: Int,
            fcDims: Array[Int],
            cinDims: Array[Int],
            mats: Array[Float],
            start: Int = 0): CINEncoder = new CINEncoder(batchSize, nFields, embeddingDim, fcDims, cinDims, mats, start)
}

