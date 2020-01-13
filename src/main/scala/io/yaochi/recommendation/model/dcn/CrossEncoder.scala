package io.yaochi.recommendation.model.dcn

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.recommendation.util.LayerUtil

import scala.collection.mutable.ArrayBuffer

class CrossEncoder(batchSize: Int,
                   nFields: Int,
                   embeddingDim: Int,
                   crossDepth: Int,
                   fcDims: Array[Int],
                   mats: Array[Float],
                   start: Int = 0) {

  private val xDim = nFields * embeddingDim

  private val shapeModule = buildShapeModule()

  private val crossLinearLayers = buildCrossLinearLayers()

  private val crossMMLayers = buildCrossMMLayers()

  private val crossAddLayers = buildAddLayers()

  private val crossBiasOffset = start + xDim * crossDepth

  private val crossBiasLayers = buildCrossBiasLayers()

  private val dnnLinearOffset = crossBiasOffset + 1 * crossDepth

  private val dnnLinearLayers = buildLinearLayers()

  private val dnnModule = buildDNNModule()

  private val outputLinearOffset = dnnLinearOffset + getDNNParameterSize

  private val outputLinearLayer = buildOutputLinearLayer()

  private val outputModule = buildOutputModule()

  private var crossInputTensors: Array[Tensor[Float]] = _

  def forward(input: Tensor[Float]): Tensor[Float] = {
    val x0Tensor = shapeModule.forward(input)
    var xkTensor = x0Tensor
    val inputTensors = ArrayBuffer[Tensor[Float]]()
    for (i <- 0 until crossDepth) {
      inputTensors += xkTensor
      val linearOutputTensor = crossLinearLayers(i).forward(xkTensor)
      val mmOutputTensor = crossMMLayers(i).forward(T(x0Tensor, linearOutputTensor))
      xkTensor = crossAddLayers(i).forward(T(mmOutputTensor, xkTensor)).toTensor[Float]
    }

    crossInputTensors = inputTensors.toArray

    val dnnOutputTensor = dnnModule.forward(x0Tensor)
    outputModule.forward(T(xkTensor, dnnOutputTensor)).toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val outputModuleInput = T.apply(
      crossAddLayers(crossDepth - 1).output.toTensor[Float],
      dnnModule.output.toTensor[Float]
    )

    val outputModuleGradTable = outputModule.backward(outputModuleInput, gradOutput).toTable
    val dnnGradTensor = dnnModule.backward(input, outputModuleGradTable[Tensor[Float]](2))
      .toTensor[Float]

    for (i <- crossDepth - 1 to 0 by -1) {
      val crossInputTensor = crossInputTensors(i - 1)
      val mmOutputTensor = crossMMLayers(i - 1).output.toTensor[Float]
      crossAddLayers(i - 1).backward(T(mmOutputTensor, crossInputTensor), gradOutput)

    }

    null
  }

  private def buildShapeModule(): Reshape[Float] = {
    Reshape(Array(batchSize, xDim), Some(false))
  }

  private def buildCrossMMLayers(): Array[MM[Float]] = {
    val layers = ArrayBuffer[MM[Float]]()
    for (i <- 0 until crossDepth) {
      layers += MM()
    }
    layers.toArray
  }

  private def buildAddLayers(): Array[Sequential[Float]] = {
    val layers = ArrayBuffer[Sequential[Float]]()
    for (i <- 0 until crossDepth) {
      layers += Sequential[Float]()
        .add(CAddTable[Float]())
        .add(crossBiasLayers(i))
    }
    layers.toArray
  }


  private def buildCrossLinearLayers(): Array[Linear[Float]] = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = start
    for (i <- 0 until crossDepth) {
      layers += LayerUtil.buildLinear(xDim, 1, mats, false, curOffset)
      curOffset += xDim * 1
    }
    layers.toArray
  }

  private def buildCrossBiasLayers(): Array[CAdd[Float]] = {
    val layers = ArrayBuffer[CAdd[Float]]()
    var curOffset = crossBiasOffset
    for (i <- 0 until crossDepth) {
      layers += LayerUtil.buildBiasLayer(1, mats, curOffset)
      curOffset += 1
    }
    layers.toArray
  }

  private def buildDNNModule(): Sequential[Float] = {
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
    var curOffset = dnnLinearOffset
    var dim = xDim
    for (fcDim <- fcDims) {
      layers += LayerUtil.buildLinear(dim, fcDim, mats, true, curOffset)
      curOffset += dim * fcDim + fcDim
      dim = fcDim
    }
    layers.toArray
  }

  private def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(JoinTable(2, 2))
      .add(outputLinearLayer)
  }

  private def buildOutputLinearLayer(): Linear[Float] = {
    val dim = xDim + fcDims.last
    LayerUtil.buildLinear(dim, 1, mats, false, outputLinearOffset)
  }

  private def getDNNParameterSize: Int = {
    val dims = Array(xDim) ++ fcDims
    (1 until dims.length)
      .map(i => nFields * dims(i - 1) * dims(i) + dims(i))
      .sum
  }
}

object CrossEncoder {
  def apply(batchSize: Int,
            nFields: Int,
            embeddingDim: Int,
            crossDepth: Int,
            fcDims: Array[Int],
            mats: Array[Float],
            start: Int = 0): CrossEncoder = new CrossEncoder(batchSize, nFields, embeddingDim, crossDepth, fcDims, mats, start)
}
